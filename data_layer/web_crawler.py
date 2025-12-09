import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import logging
from typing import List, Dict, Set, Tuple
import time
import re
import json
import xml.etree.ElementTree as ET
from collections import deque
import hashlib
from datetime import datetime
from urllib.robotparser import RobotFileParser
import random
from .data_manager import DataManager

logger = logging.getLogger(__name__)

class EnhancedWebCrawler:
    """
    Crawler nâng cao dùng cho website Arbin
    --------------------------------------------------
    - Hỗ trợ crawl toàn site (Arbin.com)
    - Tự động phát hiện sitemap, lọc URL theo robots.txt
    - Có hệ thống chấm điểm độ quan trọng (importance score)
    - Tự động nhận dạng email, số điện thoại
    - Hỗ trợ recrawl thông minh theo thời gian
    - Tích hợp lưu trữ với DataManager
    """

    def __init__(self, base_url: str = "https://www.arbin.com/"):
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.visited_urls: Set[str] = set()
        self.urls_to_visit = deque()
        self.data_manager = DataManager()
        self.data_key = "enhanced_web_crawler"
        # Tập hợp email và hotline phát hiện được
        self.emails: Set[str] = set()
        self.hotlines: Set[str] = set()
        self.crawl_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'failed_urls': []  # Lưu các URL thất bại
        }

        # Bộ nhớ theo dõi URL đã xử lý (hash nội dung + timestamp)
        self.processed_urls_info: Dict[str, Dict] = {}
        self.load_previous_crawls()

        # Thiết lập session HTTP chung để tái sử dụng kết nối
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        # Đọc và kiểm tra robots.txt
        self.robot_parser = RobotFileParser()
        self.robot_parser.set_url(urljoin(self.base_url, "robots.txt"))
        try:
            self.robot_parser.read()
        except:
            logger.warning("Could not read robots.txt")

        # Giới hạn và cấu hình crawl
        self.max_pages = 500      # Giới hạn số trang tối đa
        self.max_depth = 5        # Mức sâu tối đa khi đệ quy theo link
        self.delay = 0.5          # Thời gian nghỉ giữa các request
        self.timeout = 15         # Timeout mỗi request

        # Cấu hình số ngày recrawl cho từng loại trang
        self.recrawl_intervals = {
            'homepage': 1,
            'product_pages': 7,
            'support_pages': 30,
            'documentation': 90,
            'default': 14
        }

        # Các pattern ưu tiên crawl (liên quan tới sản phẩm / kỹ thuật)
        self.priority_patterns = [
            r'/products/', r'/software/', r'/support/', r'/resources/', r'/applications/',
            r'/technical[-_]?support/', r'/downloads?/', r'/documentation/', r'/manuals?/',
            r'/specifications?/', r'/datasheets?/', r'/brochures?/', r'/news/', r'/blog/',
            r'/articles?/', r'/tutorials?/', r'/guides?/', r'/faqs?/', r'/help/', r'/knowledge[-_]?base/'
        ]

        # Các pattern cần loại trừ khi crawl
        self.exclude_patterns = [
            r'\.(css|js)$', r'\/cart\/', r'\/checkout\/', r'\/account\/', r'\/login\/',
            r'\/register\/', r'\/wp-admin\/', r'\/wp-content\/', r'\/wp-json\/',
            r'\/wp-includes\/', r'\/cgi-bin\/', r'\?', r'\.php$', r'\.asp$', r'\.aspx$',
            r'#', r'javascript:', r'mailto:', r'tel:'
        ]

        # Tập hợp từ khóa kỹ thuật quan trọng để tính điểm nội dung
        self.important_keywords = [
            'arbin', 'battery', 'test', 'testing', 'system', 'bt-', 'lbt', 'mbt', 'mits',
            'software', 'pro', 'hardware', 'specification', 'technical', 'data', 'measurement',
            'voltage', 'current', 'capacity', 'cycler', 'tester', 'ev', 'electric', 'vehicle',
            'r&d', 'research', 'development', 'manufacturing', 'quality', 'control',
            'laboratory', 'lab', 'cell', 'lithium', 'ion', 'battery', 'analysis', 'daq',
            'acquisition', 'calibration', 'accuracy', 'resolution', 'channel', 'module', 'modular',
            'configuration'
        ]

    # -----------------------------------
    # Load dữ liệu crawl trước đó để tránh trùng lặp
    # -----------------------------------
    def load_previous_crawls(self):
        try:
            previous_data = self.data_manager.load_raw_data(self.data_key)
            if previous_data:
                for doc in previous_data:
                    url = doc.get('url')
                    if url:
                        self.processed_urls_info[url] = {
                            'content_hash': self.get_content_hash(doc.get('content', '')),
                            'crawled_at': doc.get('crawled_at', ''),
                            'title': doc.get('title', ''),
                            'content_length': len(doc.get('content', ''))
                        }
                logger.info(f"Loaded {len(self.processed_urls_info)} previously crawled URLs")
        except Exception as e:
            logger.warning(f"Could not load previous crawls: {e}")

    # -----------------------------------
    # Tiện ích xử lý URL & nội dung
    # -----------------------------------
    def get_content_hash(self, content: str) -> str:
        """Sinh mã băm MD5 để kiểm tra nội dung thay đổi"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def normalize_url(self, url: str) -> str:
        """Chuẩn hóa URL: bỏ fragment, cắt / dư thừa"""
        url = urldefrag(url)[0]
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        return parsed._replace(path=path).geturl()

    def check_robots_permission(self, url: str) -> bool:
        """Kiểm tra quyền crawl theo robots.txt"""
        return self.robot_parser.can_fetch("*", url)

    def should_crawl_url(self, url: str, parent_url: str = None) -> bool:
        """Quyết định có nên crawl URL hay không (lọc domain, loại trừ pattern xấu)"""
        url = self.normalize_url(url)
        parsed = urlparse(url)
        if parsed.netloc and parsed.netloc != self.base_domain:
            return False
        if not self.check_robots_permission(url):
            return False
        url_lower = url.lower()
        for pattern in self.exclude_patterns:
            if re.search(pattern, url_lower, re.IGNORECASE):
                return False
        return True

    # -----------------------------------
    # Logic xác định tái crawl (Recrawl)
    # -----------------------------------
    def get_recrawl_interval(self, url: str) -> int:
        """Trả về số ngày giữa 2 lần crawl cho từng loại trang"""
        url_lower = url.lower()
        if url == self.base_url:
            return self.recrawl_intervals['homepage']
        for pattern, days in [
            (r'/products/', self.recrawl_intervals['product_pages']),
            (r'/software/', self.recrawl_intervals['product_pages']),
            (r'/support/', self.recrawl_intervals['support_pages']),
            (r'/technical[-_]?support/', self.recrawl_intervals['support_pages']),
            (r'/documentation/', self.recrawl_intervals['documentation']),
            (r'/manuals?/', self.recrawl_intervals['documentation']),
        ]:
            if re.search(pattern, url_lower):
                return days
        return self.recrawl_intervals['default']

    def should_recrawl_url(self, url: str) -> Tuple[bool, str]:
        """
        Xác định có cần tái crawl URL không
        - Dựa vào ngày crawl cũ + khoảng thời gian định nghĩa
        """
        url = self.normalize_url(url)
        if url not in self.processed_urls_info:
            return True, "New URL"

        info = self.processed_urls_info[url]
        crawled_at = info.get('crawled_at', '')
        if not crawled_at:
            return True, "No previous crawl timestamp"
        try:
            prev_time = datetime.strptime(crawled_at, "%Y-%m-%d %H:%M:%S")
            days_since = (datetime.now() - prev_time).days
            recrawl_days = self.get_recrawl_interval(url)
            if days_since >= recrawl_days:
                return True, f"Due for recrawl ({days_since}/{recrawl_days} days)"
            else:
                return False, f"Recently crawled ({days_since}/{recrawl_days} days ago)"
        except Exception as e:
            logger.warning(f"Error parsing crawl time for {url}: {e}")
            return True, "Invalid timestamp format"

    # -----------------------------------
    # Trích xuất liên kết và nội dung
    # -----------------------------------
    def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[Tuple[str, str]]:
        """Lấy tất cả các liên kết hợp lệ trong trang"""
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = urljoin(current_url, a_tag['href'])
            href = self.normalize_url(href)
            if self.should_crawl_url(href, current_url):
                links.add((href, a_tag.get_text(strip=True)))
        return list(links)

    def extract_priority_content(self, soup: BeautifulSoup, url: str) -> Dict[str, any]:
        """
        Lấy phần nội dung chính (main content) từ các selector phổ biến
        Ưu tiên div/article chính, fallback là toàn bộ text
        """
        content = {}
        content_selectors = [
            ('main', 'main'), ('article', 'article'), ('div[class*="content"]', 'content_div'),
            ('div[class*="main"]', 'main_div'), ('div[class*="post"]', 'post'),
            ('div[class*="entry"]', 'entry'), ('section', 'section'),
            ('div[class*="body"]', 'body'), ('div[class*="text"]', 'text'),
            ('div[class*="description"]', 'description'), ('div[class*="wp-block-"]', 'wp_block'),
            ('div[class*="et_pb_text"]', 'et_text'), ('div[class*="et_pb_module"]', 'et_module')
        ]
        for selector, name in content_selectors:
            elements = soup.select(selector)
            for i, elem in enumerate(elements[:3]):
                text = elem.get_text(separator=' ', strip=True)
                if text and len(text) > 100:
                    content[f"{name}_{i}"] = {
                        'text': text,
                        'selector': selector,
                        'length': len(text)
                    }
        if not content:
            all_text = soup.get_text(separator=' ', strip=True)
            content['full_page'] = {'text': all_text, 'selector': 'full_page', 'length': len(all_text)}
        return content

    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, any]:
        """Trích xuất metadata như title, description, headers, link/image count"""
        metadata = {'title': '', 'description': '', 'keywords': [], 'headers': {}, 'links_count': 0, 'images_count': 0}
        if soup.title and soup.title.string:
            metadata['title'] = soup.title.string.strip()
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            metadata['description'] = meta_desc['content'].strip()
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords and meta_keywords.get('content'):
            metadata['keywords'] = [k.strip() for k in meta_keywords['content'].split(',')]
        for i in range(1, 7):
            headers = soup.find_all(f'h{i}')
            if headers:
                metadata['headers'][f'h{i}'] = [h.get_text(strip=True) for h in headers[:5]]
        metadata['links_count'] = len(soup.find_all('a', href=True))
        metadata['images_count'] = len(soup.find_all('img'))
        return metadata

    def analyze_content_importance(self, text: str, url: str) -> float:
        """
        Tính điểm độ quan trọng của nội dung
        - Dựa trên số từ khóa kỹ thuật và pattern ưu tiên
        - Cộng điểm cho nội dung dài
        """
        importance_score = 0
        text_lower = text.lower()
        for keyword in self.important_keywords:
            if keyword.lower() in text_lower:
                importance_score += 1
        url_lower = url.lower()
        for pattern in self.priority_patterns:
            if re.search(pattern, url_lower, re.IGNORECASE):
                importance_score += 3
        if len(text) > 1000:
            importance_score += 2
        elif len(text) > 500:
            importance_score += 1
        return importance_score

    # ------------------------------
    # Hàm crawl từng trang
    # ------------------------------
    def crawl_page(self, url: str, depth: int = 0, force_recrawl: bool = False, parent_url: str = None, link_text: str = "") -> Dict[str, any]:
        """
        Crawl 1 trang đơn lẻ:
        - Gọi HTTP GET
        - Làm sạch nội dung
        - Tính hash & điểm quan trọng
        - Trích xuất email, phone
        - Thêm các link mới vào hàng đợi
        """
        self.crawl_stats['total'] += 1
        url = self.normalize_url(url)
        if not force_recrawl and url in self.processed_urls_info:
            should_recrawl, reason = self.should_recrawl_url(url)
            if not should_recrawl:
                # Nếu URL gần đây đã crawl → bỏ qua (trả cached)
                info = self.processed_urls_info[url]
                return {
                    'url': url, 'title': info.get('title', ''), 'content': '', 'source': 'web',
                    'source_type': 'web', 'depth': depth, 'crawled_at': info['crawled_at'],
                    'content_length': info.get('content_length', 0), 'importance_score': 0,
                    'status_code': 304, 'cached': True, 'previous_hash': info.get('content_hash', ''),
                    'recrawl_reason': reason
                }

        if url in self.visited_urls:
            return {}

        # Nếu là file PDF → chỉ lưu metadata, không crawl nội dung
        if url.lower().endswith('.pdf'):
            self.data_manager.save_file_metadata({'url': url, 'type': 'pdf', 'source_page': parent_url})
            return {}

        logger.info(f"🔍 Crawling: {url} (depth: {depth})")
        self.visited_urls.add(url)

        # Delay ngẫu nhiên nhẹ để tránh bị chặn
        time.sleep(self.delay + random.uniform(0.1, 0.3))

        # Cơ chế retry 3 lần nếu request thất bại
        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                self.crawl_stats['successful'] += 1
                break
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
                if attempt == 2:  # Lần thử cuối cùng thất bại
                    self.crawl_stats['failed'] += 1
                    self.crawl_stats['failed_urls'].append({
                        'url': url,
                        'error': str(e),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'depth': depth,
                        'parent_url': parent_url
                    })
                time.sleep(2 ** attempt)
        else:
            logger.error(f"Failed all retries: {url}")
            return {}

        # Bỏ qua nếu không phải HTML
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            return {}

        # Dò encoding và parse HTML
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')

        # Loại bỏ các phần không cần thiết
        for selector in ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe', 'noscript', 'svg']:
            for element in soup.select(selector):
                element.decompose()

        # Trích xuất nội dung và metadata
        content_data = self.extract_priority_content(soup, url)
        metadata = self.extract_metadata(soup, url)

        # Gộp toàn bộ text lại
        combined_text = ' '.join([d['text'] for d in content_data.values()])
        cleaned_text = ' '.join(chunk.strip() for line in combined_text.splitlines() for chunk in line.split("  ") if chunk.strip())

        importance_score = self.analyze_content_importance(cleaned_text, url)
        content_hash = self.get_content_hash(cleaned_text)
        previous_hash = self.processed_urls_info.get(url, {}).get('content_hash')
        is_updated = previous_hash != content_hash if previous_hash else False

        # Trích xuất email và số điện thoại
        emails = list(set(re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", cleaned_text)))
        phones = list(set(re.findall(r"(\+?\d{1,3})?[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}", cleaned_text)))

        # Kết quả crawl 1 trang
        result = {
            'url': url,
            'title': metadata['title'],
            'content': cleaned_text,
            'source': 'web',
            'source_type': 'web',
            'depth': depth,
            'crawled_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'content_length': len(cleaned_text),
            'importance_score': importance_score,
            'status_code': response.status_code,
            'metadata': metadata,
            'content_sections': content_data,
            'content_hash': content_hash,
            'is_updated': is_updated,
            'previous_hash': previous_hash,
            'parent_url': parent_url,
            'link_text': link_text,
            'emails': emails,
            'phones': phones
        }

        # Cập nhật thông tin URL đã xử lý
        self.processed_urls_info[url] = {
            'content_hash': content_hash,
            'crawled_at': result['crawled_at'],
            'title': result['title'],
            'content_length': result['content_length']
        }

        # Nếu chưa đạt max_depth → thêm link mới vào queue
        if depth < self.max_depth:
            links = self.extract_links(soup, url)
            for link, text in links:
                if link not in self.visited_urls and link not in [item[0] for item in self.urls_to_visit]:
                    self.urls_to_visit.append((link, depth + 1, False, url, text))

        logger.info(f"Crawled: {url} - {len(cleaned_text)} chars, score: {importance_score}, updated: {is_updated}, emails: {len(emails)}, phones: {len(phones)}")
        return result

    # ------------------------------
    # Xử lý Sitemap
    # có nhiệm vụ tự động phát hiện và đọc danh sách URL có sẵn từ sitemap của website, 
    # thay vì phải tìm link thủ công bằng cách crawl từng trang một.
    # ------------------------------
    def discover_sitemaps(self) -> List[str]:
        """Tìm tất cả sitemap có thể từ robots.txt và các tên phổ biến"""
        sitemap_urls = []
        common_paths = [
            'sitemap.xml', 'sitemap_index.xml', 'sitemap1.xml',
            'sitemap-news.xml', 'sitemap-products.xml', 'sitemap-articles.xml', 'robots.txt'
        ]
        for path in common_paths:
            sitemap_url = urljoin(self.base_url, path)
            try:
                response = self.session.head(sitemap_url, timeout=5)
                if response.status_code == 200:
                    sitemap_urls.append(sitemap_url)
            except:
                continue

        # Kiểm tra robots.txt để tìm dòng "Sitemap:"
        robots_url = urljoin(self.base_url, 'robots.txt')
        try:
            response = self.session.get(robots_url, timeout=5)
            if response.status_code == 200:
                for line in response.text.splitlines():
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        sitemap_urls.append(urljoin(self.base_url, sitemap_url))
        except:
            pass
        return list(set(sitemap_urls))

    def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """Phân tích sitemap XML và trích xuất danh sách URL"""
        urls = []
        try:
            response = self.session.get(sitemap_url, timeout=10)
            if 'xml' in response.headers.get('Content-Type', ''):
                try:
                    root = ET.fromstring(response.content)
                    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                    url_tags = root.findall('.//ns:url/ns:loc', namespace) or root.findall('.//loc')
                    for url_tag in url_tags:
                        url = url_tag.text.strip()
                        if url and self.should_crawl_url(url):
                            urls.append(url)
                except ET.ParseError:
                    # Nếu XML lỗi → fallback sang BeautifulSoup
                    soup = BeautifulSoup(response.content, 'xml')
                    url_tags = soup.find_all('loc')
                    for url_tag in url_tags:
                        url = url_tag.text.strip()
                        if url and self.should_crawl_url(url):
                            urls.append(url)

            logger.info(f"Parsed {len(urls)} URLs from sitemap: {sitemap_url}")
        except Exception as e:
            logger.error(f"Error parsing sitemap {sitemap_url}: {e}")
        return urls
    
    def print_simple_statistics(self):
        """Hiển thị thống kê đơn giản về kết quả crawl"""
        print("\n" + "="*60)
        print("📊 THỐNG KÊ CRAWL")
        print("="*60)
        
        total = self.crawl_stats['total']
        if total == 0:
            print("❌ Chưa thực hiện crawl nào")
            return
            
        successful = self.crawl_stats['successful']
        failed = self.crawl_stats['failed']
        cached = total - successful - failed  # URL được cache
        
        success_rate = (successful / total * 100) if total > 0 else 0
        failure_rate = (failed / total * 100) if total > 0 else 0
        cache_rate = (cached / total * 100) if total > 0 else 0
        
        print(f"📈 TỔNG SỐ URL XỬ LÝ: {total}")
        print(f"✅ THÀNH CÔNG: {successful} ({success_rate:.1f}%)")
        print(f"❌ THẤT BẠI: {failed} ({failure_rate:.1f}%)")
        print(f"💾 ĐÃ CACHE (không thay đổi): {cached} ({cache_rate:.1f}%)")
        print()
        
        # Hiển thị các URL thất bại nếu có
        if self.crawl_stats['failed_urls']:
            print("🔴 CÁC URL THẤT BẠI:")
            for i, failed_url in enumerate(self.crawl_stats['failed_urls'][:10], 1):
                print(f"  {i}. {failed_url['url']}")
                print(f"     Lỗi: {failed_url['error'][:100]}...")
                print(f"     Độ sâu: {failed_url['depth']}")
                if failed_url['parent_url']:
                    print(f"     Từ trang: {failed_url['parent_url']}")
                print()
            
            if len(self.crawl_stats['failed_urls']) > 10:
                print(f"  ... và {len(self.crawl_stats['failed_urls']) - 10} URL thất bại khác")
        
        print("="*60)

    def get_statistics_summary(self) -> Dict[str, any]:
        """Trả về tóm tắt thống kê dạng dictionary"""
        total = self.crawl_stats['total']
        successful = self.crawl_stats['successful']
        failed = self.crawl_stats['failed']
        cached = total - successful - failed
        
        return {
            'total_urls': total,
            'successful': successful,
            'failed': failed,
            'cached': cached,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'failure_rate': (failed / total * 100) if total > 0 else 0,
            'cache_rate': (cached / total * 100) if total > 0 else 0,
            'failed_urls': self.crawl_stats['failed_urls'],
            'failed_count': len(self.crawl_stats['failed_urls'])
        }

    # -----------------------------------
    # Crawl toàn site
    # -----------------------------------
    def get_initial_urls(self, force_recrawl: bool = False) -> List[str]:
        """Lấy danh sách URL khởi tạo (ưu tiên sitemap, sau đó là các section chính)"""
        initial_urls = []
        sitemaps = self.discover_sitemaps()
        for sitemap in sitemaps:
            urls = self.parse_sitemap(sitemap)
            if not force_recrawl:
                filtered = []
                for url in urls[:100]:
                    should_recrawl, _ = self.should_recrawl_url(url)
                    if should_recrawl:
                        filtered.append(url)
                initial_urls.extend(filtered)
            else:
                initial_urls.extend(urls[:100])

        # Nếu không tìm thấy sitemap → fallback vào homepage và các section chính
        if not initial_urls:
            should_recrawl, _ = self.should_recrawl_url(self.base_url)
            if force_recrawl or should_recrawl:
                initial_urls.append(self.base_url)
            important_sections = ['/products/', '/software/', '/support/', '/resources/', '/applications/', '/about/', '/news/', '/blog/']
            for section in important_sections:
                url = urljoin(self.base_url, section)
                should_recrawl, _ = self.should_recrawl_url(url)
                if force_recrawl or should_recrawl:
                    initial_urls.append(url)

        # Chấm điểm ưu tiên theo pattern
        prioritized = []
        for url in initial_urls:
            priority = sum(2 for pattern in self.priority_patterns if re.search(pattern, url, re.IGNORECASE))
            if url in self.processed_urls_info:
                priority -= 1
            prioritized.append((url, priority))
        prioritized.sort(key=lambda x: x[1], reverse=True)
        return [u for u, _ in prioritized[:50]]

    def crawl_site(self, force_recrawl: bool = False) -> List[Dict[str, any]]:
        """
        Hàm điều phối crawl toàn bộ website
        - Quản lý queue
        - Ghi log tiến trình
        - Lưu kết quả định kỳ
        """
        self.crawl_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'failed_urls': []
        }
        logger.info(f"Starting {'forced ' if force_recrawl else 'incremental '}crawl of {self.base_url}")
        self.visited_urls.clear()
        self.urls_to_visit.clear()
        documents = []
        new_count = updated_count = cached_count = error_count = 0

        # Lấy danh sách URL bắt đầu
        initial_urls = self.get_initial_urls(force_recrawl)
        for url in initial_urls:
            self.urls_to_visit.append((url, 0, force_recrawl, None, ""))

        while self.urls_to_visit and len(documents) < self.max_pages:
            url, depth, force, parent, link_text = self.urls_to_visit.popleft()
            if url in self.visited_urls:
                continue
            try:
                doc = self.crawl_page(url, depth, force, parent, link_text)
                if not doc:
                    continue

                # Phân loại kết quả
                if doc.get('status_code') == 304:
                    cached_count += 1
                elif doc.get('content'):
                    documents.append(doc)
                    if doc.get('is_updated', False):
                        updated_count += 1
                    else:
                        new_count += 1
                else:
                    error_count += 1

                # Log tiến trình với thống kê cơ bản
                if len(documents) % 10 == 0 or len(documents) == 1:
                    total_processed = self.crawl_stats['total']
                    successful = self.crawl_stats['successful']
                    failed = self.crawl_stats['failed']
                    success_rate = (successful / total_processed * 100) if total_processed > 0 else 0
                    
                    logger.info(
                        f"📊 Progress: {len(documents)}/{self.max_pages} pages | "
                        f"Success rate: {success_rate:.1f}% | "
                        f"New: {new_count} | Updated: {updated_count} | "
                        f"Cached: {cached_count} | Errors: {error_count}"
                    )

                # Backup queue mỗi 50 trang
                if len(documents) % 50 == 0:
                    self.data_manager.save_raw_data(list(self.urls_to_visit), "crawl_queue_backup")

            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                error_count += 1
                self.crawl_stats['failed'] += 1
                self.crawl_stats['failed_urls'].append({
                    'url': url,
                    'error': str(e),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'depth': depth,
                    'parent_url': parent
                })

        # Sau khi crawl xong → lưu kết quả
        if documents:
            self.save_crawl_results(documents)

        # Hiển thị thống kê cuối cùng
        self.print_simple_statistics()
        
        # Lưu thống kê vào file
        self.save_statistics_to_file()

        logger.info(f"✅ Crawling completed. Total: {len(documents)} pages, New: {new_count}, Updated: {updated_count}, Cached: {cached_count}, Errors: {error_count}")

        
        return documents
    def save_statistics_to_file(self):
        """Lưu thống kê vào file JSON"""
        stats = self.get_statistics_summary()
        stats['base_url'] = self.base_url
        stats['crawl_completed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        filename = f"crawl_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = f"./data/inspection/{filename}"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            logger.info(f"Statistics saved to {filepath}")
        except Exception as e:
            logger.error(f"Could not save statistics: {e}")

    # -----------------------------------
    # Lưu kết quả crawl
    # -----------------------------------
    def save_crawl_results(self, documents: List[Dict[str, any]]):
        """
        Lưu toàn bộ kết quả crawl + xuất thống kê
        Bao gồm:
        - Số lượng trang mới, cập nhật, cache
        - Phân phối điểm quan trọng
        - 3 mẫu nội dung preview
        """
        try:
            self.data_manager.save_raw_data(documents, self.data_key)

            new_docs = [d for d in documents if not d.get('cached') and not d.get('is_updated')]
            updated_docs = [d for d in documents if d.get('is_updated')]
            cached_docs = [d for d in documents if d.get('cached')]

            stats = {
                'total_pages_crawled': len(documents),
                'new_documents': len(new_docs),
                'updated_documents': len(updated_docs),
                'cached_documents': len(cached_docs),
                'total_content_chars': sum(len(d.get('content', '')) for d in documents),
                'avg_content_length': (
                    sum(len(d.get('content', '')) for d in documents) / len(documents) if documents else 0
                ),
                'max_depth': max(d.get('depth', 0) for d in documents),
                'processed_urls_total': len(self.processed_urls_info),
                'importance_score_distribution': {
                    'high': len([d for d in documents if d.get('importance_score', 0) > 5]),
                    'medium': len([d for d in documents if 2 < d.get('importance_score', 0) <= 5]),
                    'low': len([d for d in documents if d.get('importance_score', 0) <= 2])
                },
                'top_new_urls': [d['url'] for d in new_docs[:5]],
                'top_updated_urls': [d['url'] for d in updated_docs[:5]]
            }

            # Mẫu preview nội dung
            sample_content = []
            for doc in documents[:3]:
                sample_content.append({
                    'url': doc['url'],
                    'title': doc.get('title', ''),
                    'status': 'cached' if doc.get('cached') else 'updated' if doc.get('is_updated') else 'new',
                    'content_preview': (doc.get('content', '')[:300] + '...')
                    if len(doc.get('content', '')) > 300 else doc.get('content', ''),
                    'importance_score': doc.get('importance_score', 0),
                    'content_length': len(doc.get('content', ''))
                })
            stats['sample_content'] = sample_content

            stats['crawl_success_rate'] = self.get_statistics_summary()
            # Xuất thống kê ra file JSON inspection
            self.data_manager.export_for_inspection(stats, "enhanced_crawling_stats", "json")

            logger.info(f"Crawling Summary:")
            logger.info(f"  - Total pages: {stats['total_pages_crawled']}")
            logger.info(f"  - New documents: {stats['new_documents']}")
            logger.info(f"  - Updated documents: {stats['updated_documents']}")
            logger.info(f"  - Cached documents: {stats['cached_documents']}")
            logger.info(f"  - Total processed URLs: {stats['processed_urls_total']}")
        except Exception as e:
            logger.error(f"Error saving crawl results: {e}")

# Giữ alias cũ để tương thích ngược
WebCrawler = EnhancedWebCrawler
