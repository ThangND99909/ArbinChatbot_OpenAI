"""
🔍 STEP 5: Testing and Validation
Kiểm tra chất lượng retrieval và validation pipeline
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from data_layer.data_manager import DataManager
from data_layer.vector_store import EnhancedVectorStore

# Configuration
CONFIG = {
    "persist_directory": "./chroma_db",
    "collection_name": "arbin_documents",
    "test_queries": [
        "battery testing systems",
        "Arbin software features",
        "technical support contact",
        "product specifications",
        "lithium ion battery testing",
        "data acquisition systems",
        "cell testing equipment",
        "battery cycler specifications",
        "Arbin MITS Pro software",
        "how to calibrate Arbin equipment"
    ],
    "k_results": 5,
    "score_threshold": 0.5,
    "log_level": logging.INFO
}

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=CONFIG["log_level"],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/testing_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def test_basic_retrieval(vector_store):
    """Test basic retrieval functionality"""
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 60)
    print("🔍 BASIC RETRIEVAL TESTS")
    print("=" * 60)
    
    test_results = {}
    
    for query in CONFIG["test_queries"]:
        print(f"\n📝 Query: '{query}'")
        
        try:
            results = vector_store.search_similar(
                query=query,
                k=CONFIG["k_results"],
                score_threshold=CONFIG["score_threshold"]
            )
            
            test_results[query] = {
                "total_results": len(results),
                "scores": [r.get("score", 0) for r in results],
                "top_score": results[0].get("score", 0) if results else 0,
                "sources": [r.get("metadata", {}).get("source", "unknown") for r in results],
                "content_previews": [r.get("text", "")[:100] + "..." for r in results[:2]]
            }
            
            if results:
                print(f"   ✅ Found {len(results)} results")
                print(f"   🏆 Top score: {results[0].get('score', 0):.3f}")
                print(f"   📍 Sources: {', '.join(set(test_results[query]['sources']))}")
                
                # Show top result preview
                top_result = results[0]
                preview = top_result.get("text", "")[:150]
                print(f"   📄 Preview: {preview}...")
            else:
                print(f"   ⚠️ No results above threshold {CONFIG['score_threshold']}")
                
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")
            print(f"   ❌ Error: {e}")
            test_results[query] = {"error": str(e)}
    
    return test_results

def test_advanced_features(vector_store):
    """Test advanced features"""
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 60)
    print("🚀 ADVANCED FEATURE TESTS")
    print("=" * 60)
    
    advanced_results = {}
    
    # Test 1: Semantic search with multiple queries
    print("\n1. Multi-query semantic search:")
    try:
        multi_results = vector_store.semantic_search(
            queries=["battery test", "cell testing", "calibration"],
            k=3,
            combine_results=True
        )
        
        advanced_results["multi_query"] = {
            "success": True,
            "total_results": len(multi_results),
            "avg_score": sum(r.get("score", 0) for r in multi_results) / len(multi_results) if multi_results else 0
        }
        
        print(f"   ✅ Combined {len(multi_results)} results from 3 queries")
        
    except Exception as e:
        logger.error(f"Error in multi-query search: {e}")
        advanced_results["multi_query"] = {"success": False, "error": str(e)}
        print(f"   ❌ Error: {e}")
    
    # Test 2: Filter by metadata
    print("\n2. Metadata filtering:")
    try:
        filtered_results = vector_store.search_similar(
            query="software",
            k=5,
            filter_metadata={"source_type": "web"},  # Only web documents
            score_threshold=0.4
        )
        
        advanced_results["metadata_filter"] = {
            "success": True,
            "total_results": len(filtered_results),
            "all_web": all(r.get("metadata", {}).get("source_type") == "web" for r in filtered_results)
        }
        
        print(f"   ✅ Found {len(filtered_results)} web documents about 'software'")
        print(f"   📊 All results are web docs: {advanced_results['metadata_filter']['all_web']}")
        
    except Exception as e:
        logger.error(f"Error in metadata filtering: {e}")
        advanced_results["metadata_filter"] = {"success": False, "error": str(e)}
        print(f"   ❌ Error: {e}")
    
    # Test 3: Collection statistics
    print("\n3. Collection statistics:")
    try:
        stats = vector_store.get_collection_stats()
        advanced_results["collection_stats"] = stats
        
        print(f"   ✅ Collection has {stats.get('total_documents', 0):,} documents")
        print(f"   📈 Source distribution: {stats.get('source_types_distribution', {})}")
        print(f"   💾 Size: {(stats.get('total_documents', 0) / stats.get('collection_size_limit', 100000) * 100):.1f}% of limit")
        
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        advanced_results["collection_stats"] = {"error": str(e)}
        print(f"   ❌ Error: {e}")
    
    # Test 4: Health check
    print("\n4. System health check:")
    try:
        health = vector_store.get_health_status()
        advanced_results["health_check"] = health
        
        print(f"   ✅ Status: {health.get('status', 'unknown')}")
        print(f"   📊 Documents: {health.get('collection_count', 0):,}")
        print(f"   🏥 Embedding model: {health.get('embedding_model', 'unknown')}")
        print(f"   💾 Storage: {health.get('storage', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        advanced_results["health_check"] = {"error": str(e)}
        print(f"   ❌ Error: {e}")
    
    return advanced_results

def generate_report(test_results, advanced_results, vector_store, data_manager):
    """Generate comprehensive test report"""
    print("\n" + "=" * 60)
    print("📊 TESTING REPORT")
    print("=" * 60)
    
    # Calculate statistics
    all_scores = []
    successful_queries = 0
    total_queries = len(test_results)
    
    for query, result in test_results.items():
        if "error" not in result and result.get("total_results", 0) > 0:
            successful_queries += 1
            all_scores.extend(result.get("scores", []))
    
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
    
    # Print summary
    print(f"\n📈 RETRIEVAL PERFORMANCE:")
    print(f"  • Total test queries: {total_queries}")
    print(f"  • Successful queries: {successful_queries} ({success_rate:.1f}%)")
    print(f"  • Average relevance score: {avg_score:.3f}")
    print(f"  • Score threshold used: {CONFIG['score_threshold']}")
    
    if all_scores:
        print(f"  • Min score: {min(all_scores):.3f}")
        print(f"  • Max score: {max(all_scores):.3f}")
    
    # Print query-by-query results
    print(f"\n🔍 QUERY RESULTS:")
    for query, result in test_results.items():
        if "error" in result:
            print(f"  ❌ '{query}': Error - {result['error']}")
        else:
            status = "✅" if result.get("total_results", 0) > 0 else "⚠️"
            print(f"  {status} '{query}': {result.get('total_results', 0)} results, top score: {result.get('top_score', 0):.3f}")
    
    # Print collection info
    stats = vector_store.get_collection_stats()
    print(f"\n🏢 COLLECTION STATUS:")
    print(f"  • Total documents: {stats.get('total_documents', 0):,}")
    print(f"  • Local documents: {stats.get('source_types_distribution', {}).get('document', 0):,}")
    print(f"  • Web documents: {stats.get('source_types_distribution', {}).get('web', 0):,}")
    print(f"  • Cache size: {stats.get('cache_size', 0)} queries")
    print(f"  • Storage usage: {stats.get('current_size_percentage', 0):.1f}%")
    
    # Save report
    report = {
        "step": "5_test_retrieval",
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "summary": {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": success_rate,
            "avg_score": avg_score,
            "min_score": min(all_scores) if all_scores else 0,
            "max_score": max(all_scores) if all_scores else 0
        },
        "query_results": test_results,
        "advanced_tests": advanced_results,
        "collection_stats": stats,
        "recommendations": generate_recommendations(test_results, stats)
    }
    
    # Export report
    data_manager.export_for_inspection(report, "step5_test_report", "json")
    
    print(f"\n💾 Report saved to: ./data/inspection/step5_test_report.json")
    
    return report

def generate_recommendations(test_results, collection_stats):
    """Generate recommendations based on test results"""
    recommendations = []
    
    # Check if collection is empty
    total_docs = collection_stats.get("total_documents", 0)
    if total_docs == 0:
        recommendations.append({
            "priority": "HIGH",
            "issue": "Collection is empty",
            "suggestion": "Run steps 1-4 to populate the vector store"
        })
        return recommendations
    
    # Check success rate
    successful = sum(1 for r in test_results.values() if "error" not in r and r.get("total_results", 0) > 0)
    total = len(test_results)
    success_rate = successful / total if total > 0 else 0
    
    if success_rate < 0.5:
        recommendations.append({
            "priority": "HIGH",
            "issue": f"Low retrieval success rate ({success_rate:.1%})",
            "suggestion": "Consider lowering score_threshold or adding more relevant documents"
        })
    
    # Check average score
    all_scores = []
    for result in test_results.values():
        if "error" not in result:
            all_scores.extend(result.get("scores", []))
    
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        if avg_score < 0.6:
            recommendations.append({
                "priority": "MEDIUM",
                "issue": f"Low average relevance score ({avg_score:.3f})",
                "suggestion": "Check embedding model quality or document preprocessing"
            })
    
    # Check collection diversity
    source_dist = collection_stats.get("source_types_distribution", {})
    if len(source_dist) < 2:
        recommendations.append({
            "priority": "MEDIUM",
            "issue": "Limited source diversity",
            "suggestion": "Add more document types (both local and web)"
        })
    
    # Check collection size
    size_limit = collection_stats.get("collection_size_limit", 100000)
    current_size = collection_stats.get("total_documents", 0)
    usage_percent = (current_size / size_limit * 100) if size_limit > 0 else 0
    
    if usage_percent > 80:
        recommendations.append({
            "priority": "MEDIUM",
            "issue": f"Collection near capacity ({usage_percent:.1f}%)",
            "suggestion": "Consider increasing max_collection_size or implementing archival"
        })
    
    return recommendations

def main():
    """Main testing pipeline"""
    logger = setup_logging()
    
    print("=" * 60)
    print("🔍 STEP 5: TESTING AND VALIDATION")
    print("=" * 60)
    
    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize managers
    logger.info("Initializing DataManager and VectorStore...")
    data_manager = DataManager()
    
    print(f"📊 Loading vector store from {CONFIG['persist_directory']}...")
    
    try:
        vector_store = EnhancedVectorStore(
            persist_directory=CONFIG["persist_directory"],
            collection_name=CONFIG["collection_name"]
        )
        
        # Check if collection exists and has documents
        stats = vector_store.get_collection_stats()
        doc_count = stats.get("total_documents", 0)
        
        if doc_count == 0:
            print("❌ Vector store is empty!")
            print("   Please run steps 1-4 first to populate the store.")
            return
        
        print(f"✅ Vector store loaded with {doc_count:,} documents")
        print(f"📋 Running {len(CONFIG['test_queries'])} test queries...")
        
        # Run basic retrieval tests
        test_results = test_basic_retrieval(vector_store)
        
        # Run advanced feature tests
        advanced_results = test_advanced_features(vector_store)
        
        # Generate and display report
        report = generate_report(test_results, advanced_results, vector_store, data_manager)
        
        # Final status
        print("\n" + "=" * 60)
        print("✅ TESTING COMPLETE")
        print("=" * 60)
        
        success_rate = report["summary"]["success_rate"]
        if success_rate >= 70:
            print("🎉 EXCELLENT: Retrieval system is working well!")
        elif success_rate >= 50:
            print("👍 GOOD: Retrieval system is functional but can be improved")
        else:
            print("⚠️ NEEDS IMPROVEMENT: Consider reviewing the recommendations")
        
        print(f"\n📊 Final metrics:")
        print(f"  • Success rate: {success_rate:.1f}%")
        print(f"  • Average score: {report['summary']['avg_score']:.3f}")
        print(f"  • Collection size: {doc_count:,} documents")
        
        if report.get("recommendations"):
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"  • [{rec['priority']}] {rec['issue']}")
                print(f"    → {rec['suggestion']}")
        
        print(f"\n🚀 Next steps:")
        print(f"  1. Review detailed report: ./data/inspection/step5_test_report.json")
        print(f"  2. Integrate with RAG pipeline or application")
        print(f"  3. Set up monitoring and regular updates")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in testing pipeline: {e}")
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()