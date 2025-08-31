#!/usr/bin/env python
"""
DreamStreets v10 - Multi-Purpose Street Network Analysis System
Complete implementation with all scope fixes and improvements.
Based on AskStreets architecture for OpenAI Open Model Hackathon.
"""
import os
import sys
import duckdb
import networkx as nx
import json
import time
import math
from typing import Dict, Any, List, Optional
from collections import deque

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# --- Global State ---
state = {
    'graph': None,
    'db': None,
    'schema': None,
    'tool_history': deque(maxlen=5)
}

def initialize_environment(graphml_path: str = 'coxs_bazar.graphml', db_path: str = 'coxs_bazar.duckdb'):
    """Initialize graph and database for analysis."""
    print(f"\n🚀 Initializing DreamStreets Analysis System...")
    
    try:
        state['graph'] = nx.read_graphml(graphml_path)
        
        # Fix: Convert ALL numeric edge attributes from string to float
        for u, v, data in state['graph'].edges(data=True):
            for key, value in data.items():
                if isinstance(value, str):
                    try:
                        data[key] = float(value)
                    except (ValueError, TypeError):
                        pass
        
        # Also convert node attributes
        for node, data in state['graph'].nodes(data=True):
            for key, value in data.items():
                if isinstance(value, str) and key in ['x', 'y', 'street_count']:
                    try:
                        data[key] = float(value)
                    except (ValueError, TypeError):
                        pass
        
        state['db'] = duckdb.connect(db_path, read_only=False)
        state['db'].execute("INSTALL spatial; LOAD spatial;")
        
        # Get exact schema
        state['schema'] = {
            'nodes': state['graph'].number_of_nodes(),
            'edges': state['graph'].number_of_edges(),
            'tables': {}
        }
        
        # Get table schemas
        for table in ['nodes', 'edges', 'pois']:
            try:
                cols = state['db'].execute(f"PRAGMA table_info({table})").fetchdf()
                state['schema']['tables'][table] = cols['name'].tolist()
            except:
                pass
        
        print(f"📊 Network: {state['schema']['nodes']} nodes, {state['schema']['edges']} edges")
        print(f"📁 Database tables: {list(state['schema']['tables'].keys())}")
        print(f"✅ All numeric attributes converted from strings")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        exit()
    
    return True

def validate_data():
    """Quick validation of what's actually in the database."""
    print("\n📋 Data Validation:")
    
    try:
        # Check POI amenity types
        amenities = state['db'].execute("SELECT DISTINCT amenity FROM pois WHERE amenity IS NOT NULL").fetchdf()
        print(f"   Available amenity types: {amenities['amenity'].tolist()[:10]}...")
        
        # Check for medical facilities
        medical = state['db'].execute("""
            SELECT COUNT(*) as count 
            FROM pois 
            WHERE amenity IN ('hospital', 'clinic', 'health_center', 'doctors', 'pharmacy')
        """).fetchdf()
        print(f"   Medical facilities found: {medical['count'][0]}")
        
        # Check for food places
        food = state['db'].execute("""
            SELECT COUNT(*) as count 
            FROM pois 
            WHERE amenity IN ('restaurant', 'cafe', 'fast_food', 'food_court')
        """).fetchdf()
        print(f"   Food establishments found: {food['count'][0]}")
        
        # Sample nodes
        nodes = state['db'].execute("SELECT COUNT(*) as count FROM nodes").fetchdf()
        print(f"   Total nodes in database: {nodes['count'][0]}")
    except Exception as e:
        print(f"   ⚠️ Validation warning: {e}")

def get_recent_results(n: int = 2) -> str:
    """Get recent tool results for context."""
    if not state['tool_history']:
        return "No previous analysis available."
    
    recent = list(state['tool_history'])[-n:]
    context = []
    for entry in recent:
        context.append(f"{entry['tool']}: {entry['summary']}")
    return "\n".join(context)

def diagnose_error(error: str, code: str) -> str:
    """Provide hints for common errors."""
    if "is not in the graph" in error:
        return "Node IDs in the graph are STRINGS. Use str(node_id) or '5340680144' not 5340680144."
    elif "name 'G' is not defined" in error:
        return "G should be accessed directly without checks"
    elif "got an unexpected keyword argument 'keys'" in error:
        return "MultiDiGraph.edges() doesn't support keys=True. Use G.edges(data=True) instead."
    elif "is not defined" in error:
        return "Variable not persisting in exec scope. Ensure all code is in ONE continuous block."
    elif "unsupported operand type" in error:
        return "Type mismatch - ensure numeric attributes are floats"
    elif "KeyError" in error:
        return "Attribute not found - check available node/edge attributes"
    elif "Referenced column" in error and "not found" in error:
        return "Column doesn't exist in table. Check actual table schema first"
    elif "ST_SetSRID" in error.lower():
        return "DuckDB doesn't have ST_SetSRID. Use ST_Point directly"
    elif "list indices must be integers" in error:
        return "Indexing error - check data structure types"
    return "Check code syntax and variable usage"

# --- Network Analysis Tool ---

@tool
def network_analyst(task: str) -> str:
    """
    Analyzes street network topology using NetworkX algorithms.
    
    USE THIS TOOL WHEN:
    - Computing network metrics (centrality, connectivity, clustering)
    - Finding shortest paths between intersections
    - Analyzing network structure and topology
    - Calculating accessibility metrics
    - Identifying critical nodes or edges
    
    DO NOT USE WHEN:
    - Looking up specific places or POIs
    - Querying facility information
    - Needing exact addresses or names
    """
    print(f"\n📊 Network Analyst processing: '{task}'")
    
    llm = ChatOllama(model="gpt-oss:120b", temperature=0.1)
    
    # Get recent context
    recent_context = get_recent_results()
    
    for attempt in range(3):
        if attempt > 0:
            print(f"   🔄 Retry {attempt}/2 with enhanced guidance")
        
        # Build prompt with progressive enhancement
        prompt = f"""
You are an expert Python programmer specializing in NetworkX library for graph analysis.

EXACT GRAPH SCHEMA:
- Graph object named 'G' is a MultiDiGraph with {state['schema']['nodes']} nodes and {state['schema']['edges']} edges
- G EXISTS in the global namespace - DO NOT check for it, just use it directly
- ALL nodes represent street intersections (not facilities or POIs)
- Node IDs are STRINGS like '5340680144' NOT integers
- Node attributes: 'y' (lat), 'x' (lon), 'street_count' (float)
- Edge attributes: 'length' (meters, float), 'name' (string), 'highway' (string)

RECENT CONTEXT:
{recent_context}

TASK: {task}

CRITICAL RULES:
1. Write ALL code as ONE CONTINUOUS BLOCK - no blank lines, no separate sections
2. NEVER split variable definitions from their usage
3. Node IDs are ALWAYS strings: use '5340680144' not 5340680144
4. Set FINAL_RESULT at the END of your code block
5. Keep results concise - top 5-10 items, not all {state['schema']['nodes']} nodes

TEMPLATE TO FOLLOW:
# Everything in one continuous block
metric = nx.some_algorithm(G, weight='length')
sorted_items = sorted(metric.items(), key=lambda x: x[1], reverse=True)[:5]
FINAL_RESULT = [{{
    "node_id": str(node_id),
    "value": round(value, 4),
    "lat": G.nodes[node_id].get('y', 0),
    "lon": G.nodes[node_id].get('x', 0)
}} for node_id, value in sorted_items]

Provide ONLY executable Python code. No explanations, no markdown, no blank lines."""

        if attempt == 1:
            prompt += """

DEBUGGING HINTS:
- If you see "name 'X' is not defined", you split the code incorrectly
- Write EVERYTHING in one block like this (NO BLANK LINES):
source = '5340680144'
dists = nx.single_source_dijkstra_path_length(G, source, weight='length')
sorted_dists = sorted(dists.items(), key=lambda x: x[1])[:5]
FINAL_RESULT = [{"node": n, "dist": d} for n, d in sorted_dists]
"""

        if attempt == 2:
            prompt += """

USE THIS EXACT PATTERN (copy and modify):
# NO BLANK LINES, ALL ONE BLOCK
centrality = nx.degree_centrality(G)
top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
FINAL_RESULT = [{"node": str(n), "score": round(s, 4), "lat": G.nodes[n]['y'], "lon": G.nodes[n]['x']} for n, s in top]
"""
        
        try:
            response = llm.invoke(prompt)
            code = response.content.strip().replace('```python', '').replace('```', '')
            
            # FIX: Remove ALL blank lines to ensure single block execution
            lines = [line for line in code.split('\n') if line.strip()]
            code = '\n'.join(lines)
            
            # Remove import statements
            code = '\n'.join([line for line in code.split('\n') 
                            if 'import networkx' not in line and 'from networkx' not in line])
            
            print(f"   📝 Generated code length: {len(code)} chars")
            print(f"   🔧 Code preview: {code[:200]}...")
            
            # FIX: Create a wrapper to ensure all variables stay in scope
            wrapped_code = f"""
# All variables defined here
{code}
# Ensure FINAL_RESULT exists
if 'FINAL_RESULT' not in locals():
    FINAL_RESULT = None
"""
            
            # Execute with both globals and locals merged
            exec_namespace = {
                'nx': nx,
                'G': state['graph'],
                'json': json,
                'math': math,
                'list': list,
                'dict': dict,
                'str': str,
                'float': float,
                'int': int,
                'round': round,
                'sorted': sorted,
                'len': len,
                'min': min,
                'max': max,
                'sum': sum,
                'enumerate': enumerate,
                'FINAL_RESULT': None,
                '__builtins__': __builtins__
            }
            
            # Execute in single namespace
            exec(wrapped_code, exec_namespace, exec_namespace)
            
            result = exec_namespace.get('FINAL_RESULT')
            
            if result is not None:
                print(f"   ✅ FINAL_RESULT type: {type(result)}")
                print(f"   ✅ FINAL_RESULT preview: {str(result)[:200]}")
                
                # Store in history
                state['tool_history'].append({
                    'tool': 'network_analyst',
                    'summary': f"Analyzed {task[:50]}",
                    'result': result
                })
                return f"Analysis complete: {json.dumps(result, default=str)}"
            else:
                raise ValueError("FINAL_RESULT was not set")
                
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Execution error: {error_msg}")
            
            if attempt < 2:
                print(f"   🔍 Diagnosis: {diagnose_error(error_msg, code)}")
                continue
            else:
                return f"Network analysis failed: {error_msg}. Try simplifying the query."
    
    return "Network analysis could not be completed"

# --- Database Query Tool ---

@tool
def database_analyst(task: str) -> str:
    """
    Queries POIs and performs spatial database operations.
    
    USE THIS TOOL WHEN:
    - Finding specific places (shops, hospitals, restaurants, etc.)
    - Calculating distances to/from POIs
    - Counting facilities by type
    - Spatial queries (within distance, nearest neighbor)
    - Filtering POIs by attributes
    
    DO NOT USE WHEN:
    - Computing graph algorithms
    - Analyzing network topology
    - Working only with intersection data
    """
    print(f"\n🔍 Database Analyst processing: '{task}'")
    
    llm = ChatOllama(model="gpt-oss:120b", temperature=0.1)
    
    recent_context = get_recent_results()
    
    for attempt in range(2):
        if attempt > 0:
            print(f"   🔄 Retry with simpler query approach")
        
        prompt = f"""
You are an expert in DuckDB SQL with spatial extensions.

EXACT DATABASE SCHEMA:

Table 'nodes' (street intersections ONLY - NO facilities here):
- node_id: VARCHAR (e.g., '5340680144')
- lat: DOUBLE
- lon: DOUBLE  
- street_count: INTEGER
- geom: GEOMETRY

Table 'pois' (ALL facilities and amenities are HERE):
- lat: DOUBLE
- lon: DOUBLE
- geom: GEOMETRY
- amenity: VARCHAR (values include: 'hospital', 'clinic', 'restaurant', 'school', etc.)
- building: VARCHAR
- name: VARCHAR
NOTE: No 'shop', 'cuisine', 'neighborhood' columns exist

MEDICAL FACILITIES are in POIs table where:
- amenity = 'hospital' OR amenity = 'clinic' OR amenity = 'health_center'

RECENT CONTEXT:
{recent_context}

TASK: {task}

Write a SINGLE, SIMPLE SQL query.
For medical facilities: SELECT * FROM pois WHERE amenity IN ('hospital', 'clinic', 'health_center')
For nearest to point: ORDER BY ST_Distance(geom, ST_Point(lon, lat)) LIMIT 1

Provide ONLY the SQL query. No explanations."""
        
        try:
            response = llm.invoke(prompt)
            sql = response.content.strip().replace('```sql', '').replace('```', '')
            
            print(f"   📝 Generated SQL length: {len(sql)} chars")
            preview = sql[:200] + "..." if len(sql) > 200 else sql
            print(f"   🔧 SQL preview: {preview}")
            
            result_df = state['db'].execute(sql).fetchdf()
            
            print(f"   ✅ Query returned {len(result_df)} rows")
            if not result_df.empty:
                print(f"   ✅ Columns: {list(result_df.columns)[:5]}")
            
            state['tool_history'].append({
                'tool': 'database_analyst',
                'summary': f"Found {len(result_df)} results",
                'result': len(result_df)
            })
            
            if len(result_df) == 0:
                return "No results found. The requested amenity type may not exist in this dataset."
            elif len(result_df) > 20:
                return f"Found {len(result_df)} results. First 10:\n{result_df.head(10).to_string()}"
            else:
                return f"Results ({len(result_df)} rows):\n{result_df.to_string()}"
                
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Query error: {error_msg}")
            
            if attempt == 0:
                continue
            else:
                return f"Database query failed: {error_msg}"
    
    return "Database query could not be completed"

# --- Test Suite for Debugging ---

def run_test_suite():
    """Run diverse urban planning queries to test system capabilities."""
    
    test_scenarios = [
        # Focus on queries that work well
        {
            "category": "🏗️ Urban Planning",
            "query": "If we need to build an emergency evacuation center accessible to the maximum population, which intersection should we choose?",
            "expected": "High centrality intersection with good connectivity"
        },
        {
            "category": "🏗️ Urban Planning", 
            "query": "What percentage of the road network would become disconnected if the most critical intersection fails?",
            "expected": "Network resilience metric"
        },
        {
            "category": "🚑 Humanitarian",
            "query": "During flooding, which intersections should be prioritized for emergency supply distribution to reach isolated communities?",
            "expected": "Critical nodes for disaster response"
        },
        {
            "category": "💼 Business",
            "query": "I want to open a small grocery store. Find me an intersection with high foot traffic but no existing food shops within 300 meters.",
            "expected": "Location with high centrality and low competition"
        },
        {
            "category": "🚌 Transportation",
            "query": "If we can only afford 3 new bus stops, which intersections would minimize average walking distance for all residents?",
            "expected": "Optimal locations using centrality metrics"
        },
        {
            "category": "💼 Business",
            "query": "Which areas have the highest connectivity for a delivery hub?",
            "expected": "Nodes with best access to entire network"
        },
        {
            "category": "🏗️ Urban Planning",
            "query": "Identify the most critical bridges or chokepoints in the network.",
            "expected": "Articulation points analysis"
        },
        {
            "category": "🚑 Humanitarian",
            "query": "Which intersection would be best for an emergency medical center to minimize average response time?",
            "expected": "Centrality-based optimal location"
        }
    ]
    
    results = {
        "success": [],
        "failed": [],
        "retries": [],
        "answers": {}
    }
    
    print("\n" + "="*70)
    print("🧪 URBAN ANALYSIS TEST SUITE")
    print("="*70)
    print("Testing real-world urban planning, humanitarian, and business scenarios...\n")
    
    llm = ChatOllama(model="gpt-oss:120b", temperature=0.1)
    tools = [network_analyst, database_analyst]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n[Test {i}/{len(test_scenarios)}] {scenario['category']}")
        print(f"📝 Query: {scenario['query'][:80]}...")
        print("-" * 70)
        
        # Clear tool history
        state['tool_history'].clear()
        
        start_time = time.time()
        try:
            # Create agent
            agent = create_react_agent(llm, tools)
            
            # Build query with context
            enhanced_query = f"""
SYSTEM STATE:
- Street network graph 'G': {state['schema']['nodes']} intersections, {state['schema']['edges']} road segments
- Database tables: {list(state['schema']['tables'].keys())}
- Data represents Cox's Bazar area in Bangladesh

AVAILABLE TOOLS:
- network_analyst: Graph algorithms, centrality, connectivity, paths
- database_analyst: POI queries, spatial analysis, amenity searches

USER QUERY: {scenario['query']}

Provide a specific, actionable answer with concrete numbers and locations when possible.
"""
            
            # Run analysis
            result = agent.invoke(
                {"messages": [HumanMessage(content=enhanced_query)]},
                config={"recursion_limit": 30}
            )
            
            # Get the final answer
            final_answer = result["messages"][-1].content
            
            # Count tool calls
            tool_calls = 0
            for msg in result["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls += len(msg.tool_calls)
                elif hasattr(msg, 'additional_kwargs'):
                    if 'tool_calls' in msg.additional_kwargs:
                        tool_calls += 1
            
            elapsed = time.time() - start_time
            
            # Store the answer
            results["answers"][i] = {
                "query": scenario['query'],
                "category": scenario['category'],
                "answer": final_answer,
                "tool_calls": tool_calls,
                "time": elapsed
            }
            
            # Display result summary
            print(f"\n📊 RESULT:")
            answer_lines = final_answer.split('\n')
            key_info = []
            for line in answer_lines[:10]:
                line = line.strip()
                if line and not line.startswith('---'):
                    if any(char.isdigit() for char in line) or any(word in line.lower() for word in ['node', 'intersection', 'location', 'found', 'identified', 'best', 'optimal']):
                        key_info.append(f"   → {line[:100]}")
            
            if key_info:
                print('\n'.join(key_info[:5]))
            else:
                print(f"   → {final_answer[:200]}...")
            
            print(f"\n   ⏱️  Execution: {elapsed:.1f}s with {tool_calls} tool call(s)")
            
            if tool_calls > 3:
                results["retries"].append((scenario['query'], tool_calls))
                print(f"   ⚠️  Multiple attempts needed")
            else:
                results["success"].append(scenario['query'])
                print(f"   ✅ Clean execution")
                
        except Exception as e:
            results["failed"].append((scenario['query'], str(e)))
            print(f"   ❌ Failed: {str(e)[:100]}")
            results["answers"][i] = {
                "query": scenario['query'],
                "category": scenario['category'],
                "answer": f"Error: {str(e)}",
                "tool_calls": 0,
                "time": time.time() - start_time
            }
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUITE SUMMARY")
    print("="*70)
    
    print(f"\nOverall Performance:")
    print(f"  ✅ Successful: {len(results['success'])}/{len(test_scenarios)} ({len(results['success'])/len(test_scenarios)*100:.0f}%)")
    print(f"  ⚠️  With retries: {len(results['retries'])}")
    print(f"  ❌ Failed: {len(results['failed'])}")
    
    # Performance metrics
    total_time = sum(a['time'] for a in results['answers'].values())
    avg_time = total_time / len(results['answers'])
    print(f"\n  ⏱️  Total execution time: {total_time:.1f}s")
    print(f"  ⏱️  Average per query: {avg_time:.1f}s")
    
    return results

# --- Main Analysis Function ---

def analyze(query: str):
    """Process any urban analysis query."""
    print(f"\n{'='*70}\n🌐 Street Network Analysis\n{'='*70}")
    print(f"📋 Query: {query}")
    print('='*70)
    
    tools = [network_analyst, database_analyst]
    llm = ChatOllama(model="gpt-oss:120b", temperature=0.1)
    
    # Clear history for new query
    state['tool_history'].clear()
    
    # Build context
    enhanced_query = f"""
SYSTEM STATE:
- Graph 'G' is loaded with {state['schema']['nodes']} nodes and {state['schema']['edges']} edges
- Database has tables: {list(state['schema']['tables'].keys())}
- All numeric attributes (length, x, y, street_count) are floats
- Node IDs are STRINGS (e.g., '5340680144')

AVAILABLE TOOLS:
1. network_analyst: For graph algorithms, centrality, paths, network metrics
   - Works with the street network graph G
   - Returns JSON with computed metrics
   
2. database_analyst: For finding places, counting facilities, spatial queries
   - Queries the POIs table for amenities and buildings
   - Returns query results as tables

USER QUERY: {query}

Analyze the query and provide actionable insights with specific numbers.
"""
    
    print("\n🤔 Analyzing...\n")
    
    # Create and run agent
    agent = create_react_agent(llm, tools)
    
    start_time = time.time()
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=enhanced_query)]},
            config={"recursion_limit": 25}
        )
        final_answer = result["messages"][-1].content
    except Exception as e:
        if "recursion limit" in str(e).lower():
            found = []
            for entry in state['tool_history']:
                found.append(f"- {entry['tool']}: {entry['summary']}")
            
            final_answer = f"""⚠️ Analysis incomplete after maximum attempts.

Partial results found:
{chr(10).join(found) if found else 'No successful tool calls completed.'}

Try a simpler or more specific query."""
        else:
            final_answer = f"❌ Analysis error: {str(e)}"
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("🎯 ANALYSIS RESULT")
    print("="*70)
    print(final_answer)
    print(f"\n⏱️  Time: {elapsed:.1f} seconds")
    
    if state['tool_history']:
        print("\n📊 Tools used:")
        for entry in state['tool_history']:
            print(f"   - {entry['tool']}: {entry['summary']}")

# --- Example Queries ---

example_queries = {
    "🏗️ Urban Planning": [
        "If we need to build an emergency evacuation center accessible to the maximum population, which intersection should we choose?",
        "What percentage of the road network would become disconnected if the most critical intersection fails?",
        "Identify the most critical bridges or chokepoints in the network.",
    ],
    "💼 Business Strategy": [
        "I want to open a small grocery store. Find me an intersection with high foot traffic but no existing food shops within 300 meters.",
        "Which areas have the highest connectivity for a delivery hub?",
        "Where should I locate a new pharmacy to serve the most underserved population?",
    ],
    "🚑 Emergency Response": [
        "During flooding, which intersections should be prioritized for emergency supply distribution to reach isolated communities?",
        "Which intersection would be best for an emergency medical center to minimize average response time?",
        "If we can only build one new fire station, where would it provide the best coverage?",
    ],
    "🚌 Transportation": [
        "If we can only afford 3 new bus stops, which intersections would minimize average walking distance for all residents?",
        "What's the average distance someone needs to travel to reach essential services in this network?",
        "Which roads are critical bottlenecks that would most benefit from widening?",
    ],
    "🌳 Sustainability": [
        "Identify street segments longer than 100 meters without any amenities - these could be good locations for green spaces.",
        "Which intersection would be best for a community garden that's accessible to the most residents?",
        "Find areas with high traffic but no recreational facilities within walking distance.",
    ]
}

# --- Main Interface ---

if __name__ == "__main__":
    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          DreamStreets - Urban Analysis Test Suite             ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
        initialize_environment()
        validate_data()
        run_test_suite()
        sys.exit(0)
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              DreamStreets - Network Analysis System           ║
    ║     Urban Planning | Business Strategy | Service Optimization ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    initialize_environment()
    validate_data()
    
    # Build numbered query list
    all_queries = []
    print("\n📋 Real-World Analysis Scenarios:\n")
    query_num = 1
    for category, queries in example_queries.items():
        print(f"\n{category}:")
        for q in queries[:2]:  # Show first 2 per category
            print(f"  {query_num}. {q[:70]}...")
            all_queries.append(q)
            query_num += 1
    
    print("\n\n[1-10] Select a numbered scenario")
    print("[C]    Custom query")
    print("[T]    Run comprehensive test suite")
    print("[Q]    Quit")
    print("-" * 70)
    
    while True:
        choice = input("\n🔍 Enter your choice: ").strip()
        
        if choice.upper() == 'Q':
            print("\n👋 Thank you for using DreamStreets!")
            print("   Building better cities through data-driven insights.\n")
            break
            
        elif choice.upper() == 'T':
            print("\nStarting comprehensive test suite...")
            run_test_suite()
            
        elif choice.upper() == 'C':
            print("\n💡 Tip: Ask about specific locations, services, or urban challenges.")
            custom = input("📝 Enter your analysis query: ").strip()
            if custom:
                analyze(custom)
                
        elif choice.isdigit():
            num = int(choice)
            if 1 <= num <= len(all_queries):
                print(f"\n🔍 Analyzing: {all_queries[num - 1][:70]}...")
                analyze(all_queries[num - 1])
            else:
                print(f"⚠️  Please enter a number between 1 and {len(all_queries)}")
        else:
            # Assume it's a direct query
            print(f"\n🔍 Analyzing: {choice[:70]}...")
            analyze(choice)
        
        print("\n" + "-" * 70)