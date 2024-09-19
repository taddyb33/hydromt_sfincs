from pathlib import Path

import os
import networkx as nx
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import box
import matplotlib.pyplot as plt
import contextily as cx
import zarr

import hydromt
from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

def main():
    sf = SfincsModel(data_libs=["/Users/taddbindas/projects/hydromt_sfincs/examples/tmp_data_catalogs/10m_huc6_lidar.yml"], root="tmp_ngwpc_data", mode="w+")
    
    gpkg = "/Users/taddbindas/hydrofabric/v20.1/gpkg/nextgen_11.gpkg"
    flowlines = gpd.read_file(gpkg, layer="flowpaths")
    nexus = gpd.read_file(gpkg, layer="nexus")
    divides = gpd.read_file(gpkg, layer="divides")
    flowpath_attributes = gpd.read_file(gpkg, layer="flowpath_attributes")
    
    G = nx.DiGraph()
    nexus_to_toid = dict(zip(nexus['id'], nexus['toid']))
    for _, node in nexus.iterrows():
        G.add_node(node['id'], 
                type=node['type'], 
                geometry=node['geometry'],
                toid=node['toid'])

    for _, edge in flowlines.iterrows():
        G.add_edge(edge['id'], edge['toid'], 
                mainstem=edge['mainstem'],
                order=edge['order'],
                hydroseq=edge['hydroseq'],
                lengthkm=edge['lengthkm'],
                areasqkm=edge['areasqkm'],
                tot_drainage_areasqkm=edge['tot_drainage_areasqkm'],
                has_divide=edge['has_divide'],
                divide_id=edge['divide_id'],
                geometry=edge['geometry'])
        
        if edge['toid'] in nexus_to_toid:
            G.add_edge(edge['toid'], nexus_to_toid[edge['toid']])

    start_node = "nex-2177032"
    end_node = "nex-2175887"
    path = nx.shortest_path(G, start_node, end_node)
    mask_flowlines = flowlines["id"].isin(path)
    mask_nexus = nexus["id"].isin(path)
    mask_divides = divides["id"].isin(path)
    mask_attributes = flowpath_attributes["id"].isin(path)

    _subset_nexus = nexus[mask_nexus]
    _subset_flowlines = flowlines[mask_flowlines]
    _subset_divides = divides[mask_divides]
    _subset_attributes = flowpath_attributes[mask_attributes]
    
    # root = Path("/Users/taddbindas/projects/hydromt_sfincs/examples/tmp_ngwpc_data/")
    # yml_str = f"""
    # meta:
    # root: {root.__str__()}
    
    # 10m_lidar:
    # path: HUC6_110701_dem.tiff
    # data_type: RasterDataset
    # driver: raster
    # driver_kwargs:
    #     chunks:
    #     x: 6000
    #     y: 6000
    # meta:
    #     category: topography
    #     crs: 5070
    # rename:
    #     10m_lidar: elevtn
    # """
    # data_lib = root / "tmp_data_catalogs/10m_huc6_lidar.yml"
    # with open(data_lib, mode="w") as f:
    #     f.write(yml_str)
        
    # Initialize SfincsModel Python class with the artifact data catalog which contains publically available data for North Italy
    # we overwrite (mode='w+') the existing model in the root directory if it exists
    # data_catalog = hydromt.DataCatalog(data_libs=[data_lib])
    
    sf.setup_grid_from_region(
        region = {'geom': '/Users/taddbindas/projects/hydromt_sfincs/examples/tmp_ngwpc_data/coffeyville/flowlines_divides.geojson'},
        res= 50,
        rotated=True,
        crs=_subset_divides.crs  # NAD83 / Conus Albers HARDCODED TODO figure out making this cleaner
    )
    datasets_dep = [{"elevtn": "10m_lidar", "zmin": 0.001}]

    # Add depth information to modelgrid based on these chosen datasets
    sf.setup_dep(datasets_dep=datasets_dep)
    
    sf.setup_mask_active(zmin=-5, reset_mask=True)
    
    # file_name = "data//compound_example_outflow_boundary_polygon.geojson"
    gdf_include = sf.data_catalog.get_geodataframe("/Users/taddbindas/projects/hydromt_sfincs/examples/tmp_ngwpc_data/coffeyville/flowlines_divides.geojson")

    # Here we add water level cells along the coastal boundary, for cells up to an elevation of -5 meters
    sf.setup_mask_bounds(btype="waterlevel", zmax=-5, reset_bounds=True)

    # Here we add outflow cells, only where clicked in shapefile along part of the lateral boundaries
    sf.setup_mask_bounds(btype="outflow", include_mask=gdf_include, reset_bounds=True)
    
    sf.setup_river_inflow(
        rivers=flowlines, keep_rivers_geom=True
    )
    
    gdf_riv = sf.geoms["rivers_inflow"].copy()
    df_reordered = pd.merge(gdf_riv["id"], flowpath_attributes, on='id', how='left')
    gdf_riv["rivwth"] = np.mean(df_reordered["TopWdth"].values) # width [m]

    # Assuming depth is 1.5m, TODO come back later
    gdf_riv["rivdph"] = 1.5  # depth [m]
    gdf_riv["manning"] = df_reordered["n"].tolist()  # manning coefficient [s.m-1/3]
    gdf_riv[["geometry", "rivwth", "manning"]]
    
    datasets_riv = [{"centerlines": gdf_riv}]
    
    # 1. rasterize the manning value of gdf to the  model grid and use this as a manning raster
    gdf_riv_buf = gdf_riv.assign(geometry=gdf_riv.geometry.buffer(0.2))
    # da_manning = sf.grid.raster.rasterize(gdf_riv_buf, "manning", nodata=np.nan)
    da_manning = sf.grid.raster.rasterize(gdf_riv, "manning", nodata=np.nan)

    # use the river manning raster in combination with vito land to derive the manning roughness file
    # NOTE that we can combine in-memory data with data from the data catalog
    datasets_rgh = [{"manning": da_manning}]
    
    sf.setup_subgrid(
        datasets_rgh=datasets_rgh,
        datasets_dep=datasets_dep,
        datasets_riv=datasets_riv,
        nr_subgrid_pixels=5,
        write_dep_tif=False,
        write_man_tif=False,
    )

if __name__ == "__main__":
    main()
