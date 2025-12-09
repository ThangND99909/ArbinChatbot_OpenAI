#!/usr/bin/env python3
"""
📊 CRAWL PROGRESS REPORT GENERATOR
Generate and display detailed crawl progress report
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.append('.')

from data_layer.data_manager import DataManager
from data_layer.web_crawler import EnhancedWebCrawler

def display_progress_summary(progress):
    """Display progress summary with visual bar"""
    summary = progress["summary"]
    
    print("\n" + "="*80)
    print("📊 CRAWL PROGRESS SUMMARY")
    print("="*80)
    
    # Create ASCII progress bar
    bar_length = 60
    crawled_width = int(summary['crawled_percent'] / 100 * bar_length)
    failed_width = int(summary['failed_percent'] / 100 * bar_length)
    skipped_width = int(summary['skipped_percent'] / 100 * bar_length)
    pending_width = bar_length - crawled_width - failed_width - skipped_width
    
    bar = ("█" * crawled_width + 
           "▒" * failed_width + 
           "░" * skipped_width + 
           " " * pending_width)
    
    print(f"\nProgress: [{bar}]")
    
    # Statistics table
    print("\n" + "-"*80)
    print(f"{'STATUS':<15} {'PERCENT':<10} {'COUNT':<10} {'DETAILS':<40}")
    print("-"*80)
    
    stats = [
        ("✅ Crawled", summary['crawled_percent'], summary['total_crawled'], "Successfully crawled"),
        ("❌ Failed", summary['failed_percent'], summary['total_failed'], "Failed to crawl"),
        ("⚠️ Skipped", summary['skipped_percent'], summary['total_skipped'], "Excluded by rules"),
        ("⬜ Pending", summary['pending_percent'], summary['total_pending'], "Not yet crawled")
    ]
    
    for status, percent, count, details in stats:
        print(f"{status:<15} {percent:>8.1f}% {count:>10,} {details:<40}")
    
    print("-"*80)
    print(f"{'🎯 Success Rate':<15} {summary['success_rate']:>8.1f}%")
    
    # Duration
    if 'crawl_duration' in progress:
        hours = progress['crawl_duration'] / 3600
        print(f"{'⏱️ Duration':<15} {hours:>8.1f} hours")

def display_uncovered_areas(uncovered):
    """Display uncovered areas"""
    if not uncovered.get("by_section"):
        print("\n✅ All important sections have been crawled!")
        return
    
    print("\n" + "="*80)
    print("🔍 UNCOVERED AREAS")
    print("="*80)
    
    total_uncovered = 0
    
    for section, urls in uncovered["by_section"].items():
        if urls:
            total_uncovered += len(urls)
            print(f"\n📁 {section.upper()} ({len(urls)} URLs):")
            
            for url_info in urls[:5]:  # Show first 5
                if isinstance(url_info, dict):
                    url = url_info.get('url', 'N/A')
                    priority = url_info.get('priority', 0)
                    reason = url_info.get('reason', 'unknown')
                    print(f"   • {url}")
                    print(f"     Priority: {priority:.1f} | Reason: {reason}")
                else:
                    print(f"   • {url_info}")
            
            if len(urls) > 5:
                print(f"   • ... and {len(urls) - 5} more URLs")
    
    print(f"\n📈 Total uncovered URLs: {total_uncovered}")

def display_pending_urls(pending_info):
    """Display pending URLs"""
    if not pending_info.get("important"):
        print("\n✅ No high-priority URLs pending!")
        return
    
    print("\n" + "="*80)
    print("🚨 HIGH-PRIORITY PENDING URLS")
    print("="*80)
    
    important = pending_info["important"]
    
    print(f"\nTotal high-priority pending: {len(important)}")
    print("\nTop 20 high-priority URLs:")
    
    for i, item in enumerate(important[:20], 1):
        print(f"\n{i:2d}. Priority: {item['priority']:.1f}")
        print(f"    URL: {item['url']}")
        print(f"    Reason: {item['reason']}")

def display_section_coverage(section_coverage):
    """Display section coverage"""
    if not section_coverage:
        return
    
    print("\n" + "="*80)
    print("📁 SECTION COVERAGE ANALYSIS")
    print("="*80)
    
    print(f"\n{'SECTION':<20} {'CRAWLED':<10} {'TOTAL':<10} {'COVERAGE':<10}")
    print("-" * 50)
    
    for section, data in section_coverage.items():
        print(f"{section:<20} {data['crawled']:<10} {data['total']:<10} {data['coverage_percent']:>8.1f}%")

def save_report(report, data_manager):
    """Save report to multiple formats"""
    # Save detailed JSON report
    data_manager.export_for_inspection(report, "crawl_progress_report", "json")
    
    # Save summary as text
    summary_text = generate_text_summary(report)
    data_manager.export_for_inspection(summary_text, "crawl_summary", "txt")
    
    # Save to reports directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f"crawl_report_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed report saved to: {report_file}")
    return report_file

def generate_text_summary(report):
    """Generate text summary of the report"""
    progress = report["progress_summary"]
    uncovered = report.get("uncovered_areas", {})
    
    text = f"""CRAWL PROGRESS REPORT
Generated: {datetime.now().isoformat()}
Base URL: {report['metadata']['base_url']}

PROGRESS SUMMARY:
✅ Crawled: {progress['crawled_percent']:.1f}% ({progress['total_crawled']:,} URLs)
❌ Failed: {progress['failed_percent']:.1f}% ({progress['total_failed']:,} URLs)
⚠️ Skipped: {progress['skipped_percent']:.1f}% ({progress['total_skipped']:,} URLs)
⬜ Pending: {progress['pending_percent']:.1f}% ({progress['total_pending']:,} URLs)
🎯 Success Rate: {progress['success_rate']:.1f}%

UNCOVERED AREAS:
"""
    
    if uncovered.get("by_section"):
        for section, urls in uncovered["by_section"].items():
            text += f"\n{section.upper()}: {len(urls)} URLs not crawled"
            for url_info in urls[:3]:
                if isinstance(url_info, dict):
                    text += f"\n  • {url_info.get('url')}"
                else:
                    text += f"\n  • {url_info}"
    
    # Add recommendations
    if report.get("recommendations"):
        text += "\n\nRECOMMENDATIONS:\n"
        for i, rec in enumerate(report["recommendations"], 1):
            text += f"{i}. {rec}\n"
    
    return text

def main():
    """Main function to generate crawl report"""
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("📊 CRAWL PROGRESS REPORT GENERATOR")
    print("="*80)
    
    # Initialize
    data_manager = DataManager()
    
    # Try to load existing crawler state or create new
    crawler = EnhancedWebCrawler(base_url="https://www.arbin.com/")
    
    print("\n📈 Generating crawl progress report...")
    
    # Generate report
    report = crawler.generate_crawl_report()
    
    # Display reports
    display_progress_summary(report["progress_summary"])
    
    if "section_coverage" in report.get("coverage_analysis", {}):
        display_section_coverage(report["coverage_analysis"]["section_coverage"])
    
    display_uncovered_areas(report.get("uncovered_areas", {}))
    
    if "pending_urls" in report:
        display_pending_urls(report["pending_urls"])
    
    # Display recommendations
    if report.get("recommendations"):
        print("\n" + "="*80)
        print("💡 RECOMMENDATIONS")
        print("="*80)
        
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i:2d}. {rec}")
    
    # Display next steps
    if report.get("next_steps"):
        print("\n" + "="*80)
        print("🚀 NEXT STEPS")
        print("="*80)
        
        for step in report["next_steps"]:
            print(f"\n📌 {step['action'].upper()} ({step['priority']} priority):")
            print(f"   {step['description']}")
            if step.get('urls_count'):
                print(f"   URLs to crawl: {step['urls_count']}")
            if step.get('estimated_time_minutes'):
                print(f"   Estimated time: {step['estimated_time_minutes']} minutes")
    
    # Save report
    report_file = save_report(report, data_manager)
    
    print("\n" + "="*80)
    print("✅ REPORT GENERATION COMPLETE")
    print("="*80)
    
    print(f"\n📁 Files saved:")
    print(f"   • JSON report: {report_file}")
    print(f"   • Inspection copy: ./data/inspection/crawl_progress_report.json")
    print(f"   • Text summary: ./data/inspection/crawl_summary.txt")
    
    print(f"\n📊 Key metrics:")
    print(f"   • Crawl coverage: {report['progress_summary']['crawled_percent']:.1f}%")
    print(f"   • Success rate: {report['progress_summary']['success_rate']:.1f}%")
    print(f"   • Pending URLs: {report['progress_summary']['total_pending']:,}")
    
    if report.get('uncovered_areas', {}).get('by_section'):
        uncovered_sections = list(report['uncovered_areas']['by_section'].keys())
        print(f"   • Uncovered sections: {', '.join(uncovered_sections)}")

if __name__ == "__main__":
    main()