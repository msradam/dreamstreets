#!/usr/bin/env python
"""
Cox's Bazar Area - Basic OpenStreetMap Data Setup
Simple extraction of street network and POIs for analysis
"""
import os
os.environ['NX_CUGRAPH_AUTOCONFIG'] = 'True'

import duckdb
import osmnx as ox
import pandas as pd
from datetime import datetime

def setup_coxs_bazar_db():
    """Load Cox's Bazar area street network and POIs from OpenStreetMap."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           Cox's Bazar Area - OpenStreetMap Data Setup        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Cox's Bazar area center point
    center = (21.2167, 92.1667)
    dist = 500  # 0.5km radius
    
    print(f"\n📍 Location: Cox's Bazar, Bangladesh")
    print(f"📏 Coverage: {dist}m radius from center")
    print(f"🌐 Coordinates: {center[0]}°N, {center[1]}°E")
    
    # Download street network
    print("\n⏳ Downloading street network from OpenStreetMap...")
    try:
        G = ox.graph_from_point(center, dist=dist, network_type="all")
        print(f"✅ Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"⚠️  Trying alternative network type...")
        G = ox.graph_from_point(center, dist=dist, network_type="all_private")
        print(f"✅ Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Check GPU acceleration
    try:
        import nx_cugraph
        print(f"✅ GPU acceleration: ENABLED (nx-cugraph {nx_cugraph.__version__})")
    except ImportError:
        print("⚠️  GPU acceleration: Not available")
    
    # Save graph
    ox.save_graphml(G, "coxs_bazar.graphml")
    print(f"💾 Graph saved to coxs_bazar.graphml")
    
    # Download POIs
    print("\n⏳ Downloading points of interest...")
    
    # Basic OSM tags for POIs
    tags = {
        'amenity': True,
        'building': True,
        'healthcare': True,
        'office': True,
        'shop': True,
        'tourism': True,
        'leisure': True,
        'name': True
    }
    
    try:
        pois_gdf = ox.features_from_point(center, tags, dist=dist)
        print(f"✅ POIs downloaded: {len(pois_gdf)} features")
    except Exception as e:
        print(f"⚠️  Simplified tag set due to: {e}")
        simple_tags = {'amenity': True, 'building': True}
        pois_gdf = ox.features_from_point(center, simple_tags, dist=dist)
        print(f"✅ POIs downloaded: {len(pois_gdf)} features")
    
    # Setup database
    print("\n🗄️ Setting up database...")
    con = duckdb.connect('coxs_bazar.duckdb')
    con.execute("INSTALL spatial; LOAD spatial;")
    
    # Drop existing tables
    for table in ['nodes', 'edges', 'pois']:
        con.execute(f"DROP TABLE IF EXISTS {table}")
    
    # Process nodes
    print("📊 Processing nodes...")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    
    nodes_df = pd.DataFrame({
        'node_id': nodes_gdf.index.astype(str),
        'lat': nodes_gdf['y'],
        'lon': nodes_gdf['x'],
        'street_count': nodes_gdf.get('street_count', 0)
    })
    
    con.execute("""
        CREATE TABLE nodes AS
        SELECT *, 
               ST_Point(lon, lat) AS geom
        FROM nodes_df
    """)
    
    # Process edges
    print("📊 Processing edges...")
    edges_df = edges_gdf.reset_index()[['u', 'v', 'length', 'name', 'highway']].fillna('')
    edges_df['u'] = edges_df['u'].astype(str)
    edges_df['v'] = edges_df['v'].astype(str)
    
    con.execute("CREATE TABLE edges AS SELECT * FROM edges_df")
    
    # Process POIs
    print("📊 Processing POIs...")
    
    # Extract coordinates (suppress CRS warning as it's fine for our use)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pois_gdf['lat'] = pois_gdf.geometry.centroid.y
        pois_gdf['lon'] = pois_gdf.geometry.centroid.x
    
    # Create basic POI dataframe with just coordinates
    pois_df = pd.DataFrame({
        'lat': pois_gdf['lat'],
        'lon': pois_gdf['lon']
    })
    
    # Add any existing OSM tags
    for col in pois_gdf.columns:
        if col not in ['geometry', 'lat', 'lon'] and not col.startswith('nodes'):
            try:
                # Try to add the column, skip if it causes issues
                pois_df[col] = pois_gdf[col].fillna('') if hasattr(pois_gdf[col], 'fillna') else pois_gdf[col]
            except:
                pass
    
    con.execute("""
        CREATE TABLE pois AS
        SELECT *, 
               ST_Point(lon, lat) AS geom
        FROM pois_df
    """)
    
    # Create spatial indexes
    print("\n🔧 Creating spatial indexes...")
    con.execute("CREATE INDEX nodes_geom_idx ON nodes USING RTREE (geom)")
    con.execute("CREATE INDEX pois_geom_idx ON pois USING RTREE (geom)")
    
    # Summary statistics
    print("\n" + "="*60)
    print("📊 DATA SUMMARY")
    print("="*60)
    
    # Network stats
    node_count = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edge_count = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    poi_count = con.execute("SELECT COUNT(*) FROM pois").fetchone()[0]
    
    print(f"\n🗺️ NETWORK:")
    print(f"   Nodes (intersections): {node_count}")
    print(f"   Edges (paths/roads): {edge_count}")
    print(f"   POIs (places): {poi_count}")
    
    # Edge types
    edge_types = con.execute("""
        SELECT highway, COUNT(*) as count 
        FROM edges 
        WHERE highway != ''
        GROUP BY highway 
        ORDER BY count DESC 
        LIMIT 5
    """).fetchdf()
    
    if not edge_types.empty:
        print(f"\n🛤️ TOP ROAD TYPES:")
        for _, row in edge_types.iterrows():
            print(f"   {row['highway']}: {row['count']}")
    
    # POI types
    print(f"\n📍 POI CATEGORIES:")
    
    # Check which columns exist and count non-empty values
    for col in ['amenity', 'healthcare', 'shop', 'tourism', 'leisure', 'building']:
        try:
            count = con.execute(f"SELECT COUNT(*) FROM pois WHERE {col} != '' AND {col} IS NOT NULL").fetchone()[0]
            if count > 0:
                print(f"   {col.capitalize()}: {count}")
        except:
            pass
    
    # Named places
    try:
        named_pois = con.execute("""
            SELECT * FROM pois 
            WHERE (name != '' AND name IS NOT NULL)
            LIMIT 5
        """).fetchdf()
        
        if not named_pois.empty:
            print(f"\n🏢 SAMPLE NAMED PLACES:")
            for _, row in named_pois.iterrows():
                # Find a type for this place
                place_type = 'location'
                for col in ['amenity', 'shop', 'tourism', 'healthcare']:
                    if col in row and row[col] and row[col] != '':
                        place_type = row[col]
                        break
                print(f"   {row['name']} ({place_type})")
    except:
        pass
    
    print(f"\n📝 DATA SOURCE:")
    print(f"   OpenStreetMap (community-mapped)")
    print(f"   Downloaded: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"   Coverage: {dist}m radius from center")
    
    print("\n" + "="*60)
    print("✅ Database 'coxs_bazar.duckdb' created successfully!")
    print("✅ Graph saved to 'coxs_bazar.graphml'")
    print("="*60)
    print("\n💡 Ready for GPT-OSS to generate custom analysis algorithms!")
    
    # Close connection
    con.close()
    
    return True

if __name__ == "__main__":
    setup_coxs_bazar_db()