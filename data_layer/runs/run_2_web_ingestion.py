#Crawl toàn bộ nội dung trang https://www.arbin.com/
import logging
import json
from data_layer.data_manager import DataManager
from data_layer.web_crawler import EnhancedWebCrawler  # đảm bảo file web_crawler.py của bạn nằm cùng data_layer/

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("🌐 [5] Starting web ingestion from Arbin website...")

    # 1️⃣ Khởi tạo DataManager và WebCrawler
    data_manager = DataManager()
    crawler = EnhancedWebCrawler(base_url="https://www.arbin.com/")

    # 2️⃣ Crawl website (incremental hoặc force full)
    force_full = False  # đổi True nếu bạn muốn crawl lại toàn bộ
    documents = crawler.crawl_site(force_recrawl=force_full)

    # 3️⃣ Kiểm tra kết quả
    print(f"\n📋 KẾT QUẢ TỔNG HỢP:")
    print(f"✅ Crawled {len(documents)} web pages thành công.")
    
    # Hiển thị thống kê từ crawler
    stats = crawler.get_statistics_summary()
    print(f"📊 Tỷ lệ thành công: {stats['success_rate']:.1f}%")
    print(f"📊 Tỷ lệ thất bại: {stats['failure_rate']:.1f}%")
    
    if stats['failed_urls']:
        print(f"🔴 Số URL thất bại: {stats['failed_count']}")
        print("📝 Danh sách URL thất bại (5 cái đầu):")
        for i, failed in enumerate(stats['failed_urls'][:5], 1):
            print(f"  {i}. {failed['url']}")
            print(f"     Lỗi: {failed['error'][:80]}...")
    
    if not documents:
        print("⚠️ No new documents found — site may be up-to-date.")
    else:
        # 4️⃣ Lưu dữ liệu raw
        data_manager.save_raw_data(documents, "web_arbin")
        data_manager.save_document_metadata(documents, "web_arbin")

        # 5️⃣ Lưu thống kê cơ bản
        stats = {
            "total_pages": len(documents),
            "timestamp": documents[0].get("crawled_at") if documents else None,
            "sample_urls": [d["url"] for d in documents[:5]],
            "source": "arbin.com",
        }

        data_manager.export_for_inspection(stats, "web_arbin_stats", "json")
        print(json.dumps(stats, indent=2, ensure_ascii=False))

        

    print("📂 Web crawl data saved in ./data/raw/ and ./data/inspection/")
