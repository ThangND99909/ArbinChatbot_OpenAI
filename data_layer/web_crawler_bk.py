import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import logging
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
import time
import re
import json
import xml.etree.ElementTree as ET
from collections import deque, defaultdict
import hashlib
from datetime import datetime, timedelta
from urllib.robotparser import RobotFileParser
import random
import asyncio
import aiohttp
import sqlite3
from contextlib import contextmanager
import pickle
import gzip
import psutil
import chardet
from langdetect import detect, LangDetectException
from difflib import SequenceMatcher
import numpy as np
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Import configuration
try:
    from crawler_config import (
        CRAWLER_CONFIG, 
        URL_PATTERNS,
        TECH_KEYWORDS,
        USER_AGENTS,
        RECRAWL_INTERVALS,
        ANTI_BOT_PATTERNS,
        SELENIUM_CONFIG
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    CRAWLER_CONFIG = {}
    URL_PATTERNS = {}
    TECH_KEYWORDS = []
    USER_AGENTS = []
    RECRAWL_INTERVALS = {}
    ANTI_BOT_PATTERNS = []
    SELENIUM_CONFIG = {}

logger = logging.getLogger(__name__)


class CrawlerEvents:
    """Event system for crawler hooks"""
    
    def __init__(self):
        self.on_url_discovered = []
        self.on_page_crawled = []
        self.on_error = []
        self.on_complete = []
        self.on_anti_bot_detected = []
        self.on_cache_hit = []
        self.on_health_check = []
    
    def register_handler(self, event: str, handler: Callable):
        """Register event handler"""
        event_map = {
            'url_discovered': self.on_url_discovered,
            'page_crawled': self.on_page_crawled,
            'error': self.on_error,
            'complete': self.on_complete,
            'anti_bot_detected': self.on_anti_bot_detected,
            'cache_hit': self.on_cache_hit,
            'health_check': self.on_health_check
        }
        
        if event in event_map:
            event_map[event].append(handler)
        else:
            logger.warning(f"Unknown event type: {event}")
    
    def trigger(self, event: str, data: Any):
        """Trigger event handlers"""
        handlers = getattr(self, f'on_{event}', [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error in event handler {event}: {e}")


class ConfigDrivenRateLimiter:
    """Rate limiter driven by configuration"""
    
    def __init__(self, config: Dict = None):
        self.config = config or CRAWLER_CONFIG.get('timing', {})
        self.base_delay = self.config.get('base_request_delay', 0.5)
        self.min_delay = self.config.get('min_delay', 0.3)
        self.max_delay = self.config.get('max_delay', 5.0)
        self.jitter_factor = self.config.get('jitter_factor', 0.2)
        
        self.timestamps = deque(maxlen=1000)
        self.lock = threading.RLock()
        self.backoff_count = 0
        self.consecutive_errors = 0
        
        # Anti-bot tracking
        self.anti_bot_detections = 0
        self.last_anti_bot_time = None
    
    def get_safe_delay(self) -> float:
        """Calculate safe delay based on current state"""
        base = self.base_delay
        
        # Apply backoff for errors
        if self.backoff_count > 0:
            backoff_factor = min(2.0, 1.0 + (self.backoff_count * 0.5))
            base *= backoff_factor
        
        # Apply anti-bot penalty
        if self.anti_bot_detections > 0:
            anti_bot_factor = 1.0 + (self.anti_bot_detections * 0.3)
            base *= anti_bot_factor
        
        # Apply jitter
        jitter = random.uniform(-self.jitter_factor, self.jitter_factor)
        delay = base * (1.0 + jitter)
        
        # Enforce limits
        delay = max(self.min_delay, min(delay, self.max_delay))
        
        return delay
    
    def wait(self):
        """Wait appropriate delay"""
        delay = self.get_safe_delay()
        time.sleep(delay)
        self.timestamps.append(time.time())
    
    def record_error(self):
        """Record error for backoff calculation"""
        self.consecutive_errors += 1
        self.backoff_count = min(self.backoff_count + 1, 10)
    
    def record_success(self):
        """Record success to reduce backoff"""
        self.consecutive_errors = 0
        if self.backoff_count > 0:
            self.backoff_count -= 1
    
    def record_anti_bot(self):
        """Record anti-bot detection"""
        self.anti_bot_detections += 1
        self.last_anti_bot_time = time.time()
    
    def reset_anti_bot(self):
        """Reset anti-bot counter"""
        self.anti_bot_detections = 0
    
    def should_skip_due_to_errors(self) -> bool:
        """Check if should skip due to too many errors"""
        skip_config = CRAWLER_CONFIG.get('skip_after', {})
        max_consecutive = skip_config.get('consecutive_errors', 5)
        
        return self.consecutive_errors >= max_consecutive


class ConfigDrivenCrawlCache:
    """SQLite-based crawl cache with config-driven settings"""
    
    def __init__(self, config: Dict = None):
        self.config = config or CRAWLER_CONFIG.get('cache', {})
        self.enabled = self.config.get('enabled', True)
        
        if not self.enabled:
            logger.info("Cache is disabled")
            return
        
        self.db_path = Path(self.config.get('db_path', 'crawler_cache.db'))
        self.cleanup_days = self.config.get('cleanup_days', 90)
        self.max_cache_size_mb = self.config.get('max_cache_size_mb', 1024)
        
        self._init_db()
        self._cleanup_if_needed()
    
    def _init_db(self):
        """Initialize database schema"""
        if not self.enabled:
            return
            
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS crawl_cache (
                    url TEXT PRIMARY KEY,
                    content_hash TEXT,
                    crawled_at TIMESTAMP,
                    title TEXT,
                    content_length INTEGER,
                    importance_score REAL,
                    metadata TEXT,
                    language TEXT,
                    encoding TEXT,
                    redirect_url TEXT,
                    section TEXT,
                    depth INTEGER,
                    status_code INTEGER,
                    last_modified TEXT,
                    etag TEXT,
                    crawl_duration REAL
                )
            """)
            
            # Create indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crawled_at 
                ON crawl_cache(crawled_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash 
                ON crawl_cache(content_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance 
                ON crawl_cache(importance_score)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_section 
                ON crawl_cache(section)
            """)
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        if not self.enabled:
            yield None
            return
            
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get(self, url: str) -> Optional[Dict]:
        """Get cached data for URL"""
        if not self.enabled:
            return None
            
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM crawl_cache WHERE url = ?",
                (url,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None
    
    def set(self, url: str, data: Dict):
        """Set cache data for URL"""
        if not self.enabled:
            return
            
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO crawl_cache 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                url,
                data.get('content_hash'),
                data.get('crawled_at'),
                data.get('title', ''),
                data.get('content_length', 0),
                data.get('importance_score', 0),
                json.dumps(data.get('metadata', {})),
                data.get('language', 'unknown'),
                data.get('encoding', 'utf-8'),
                data.get('redirect_url', ''),
                data.get('section', 'unknown'),
                data.get('depth', 0),
                data.get('status_code', 0),
                data.get('last_modified', ''),
                data.get('etag', ''),
                data.get('crawl_duration', 0)
            ))
    
    def should_recrawl(self, url: str, cached_data: Dict) -> bool:
        """Check if URL should be recrawled based on config"""
        if not self.enabled:
            return True
            
        # Get recrawl interval from config
        intervals = RECRAWL_INTERVALS
        url_lower = url.lower()
        
        # Determine URL type for recrawl interval
        url_type = 'default'
        for pattern, interval_type in [
            (r'^https?://[^/]+/?$', 'homepage'),
            (r'/products?/', 'product_pages'),
            (r'/software?/', 'software_pages'),
            (r'/support?/', 'support_pages'),
            (r'/documentation?/', 'documentation'),
            (r'/news?/', 'news_blog'),
            (r'/blog?/', 'news_blog'),
            (r'/resources?/', 'resources'),
            (r'/about/', 'about_contact'),
            (r'/contact/', 'about_contact'),
        ]:
            if re.search(pattern, url_lower):
                url_type = interval_type
                break
        
        recrawl_days = intervals.get(url_type, intervals.get('default', 14))
        
        # Check cache age
        cached_at = datetime.fromisoformat(cached_data['crawled_at'])
        cache_age = (datetime.now() - cached_at).days
        
        return cache_age >= recrawl_days
    
    def cleanup_old(self):
        """Cleanup old cache entries based on config"""
        if not self.enabled:
            return
            
        with self.get_connection() as conn:
            cutoff = (datetime.now() - timedelta(days=self.cleanup_days)).isoformat()
            conn.execute(
                "DELETE FROM crawl_cache WHERE crawled_at < ?",
                (cutoff,)
            )
            deleted = conn.total_changes
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old cache entries")
    
    def _cleanup_if_needed(self):
        """Cleanup if cache size exceeds limit"""
        if not self.enabled:
            return
            
        cache_size_mb = self.db_path.stat().st_size / 1024 / 1024
        if cache_size_mb > self.max_cache_size_mb:
            logger.warning(f"Cache size {cache_size_mb:.2f}MB exceeds limit {self.max_cache_size_mb}MB")
            self.cleanup_old()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        if not self.enabled:
            return {'enabled': False}
            
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as total FROM crawl_cache")
            total = cursor.fetchone()['total']
            
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT content_hash) as unique_count FROM crawl_cache"
            )
            unique_count = cursor.fetchone()['unique_count']
            
            cursor = conn.execute(
                "SELECT MIN(crawled_at) as oldest, MAX(crawled_at) as newest FROM crawl_cache"
            )
            dates = cursor.fetchone()
            
            cursor = conn.execute(
                "SELECT AVG(content_length) as avg_length FROM crawl_cache"
            )
            avg_length = cursor.fetchone()['avg_length']
            
            cursor = conn.execute(
                "SELECT section, COUNT(*) as count FROM crawl_cache GROUP BY section"
            )
            sections = {row['section']: row['count'] for row in cursor.fetchall()}
        
        return {
            'enabled': True,
            'total_entries': total,
            'unique_content': unique_count,
            'oldest_entry': dates['oldest'],
            'newest_entry': dates['newest'],
            'avg_content_length': avg_length,
            'sections': sections,
            'database_size_mb': self.db_path.stat().st_size / 1024 / 1024
        }


class ConfigDrivenCrawlerMonitor:
    """Monitor crawler health and performance with config-driven thresholds"""
    
    def __init__(self, config: Dict = None):
        self.config = config or CRAWLER_CONFIG.get('monitoring', {})
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        
        self.metrics_history = deque(maxlen=100)
        self.start_time = time.time()
        self.alerts = []
        self.health_check_interval = self.config.get('health_check_interval', 60)
        self.last_health_check = time.time()
    
    def update_metrics(self, crawler) -> Dict[str, Any]:
        """Update and return current metrics"""
        process = psutil.Process()
        
        metrics = {
            'performance': {
                'urls_visited': len(crawler.visited_urls),
                'queue_size': len(crawler.urls_to_visit),
                'success_count': crawler.success_count,
                'error_count': crawler.error_count,
                'error_rate': (crawler.error_count / (crawler.success_count + crawler.error_count) 
                              if (crawler.success_count + crawler.error_count) > 0 else 0),
                'success_rate': (crawler.success_count / (crawler.success_count + crawler.error_count) 
                                if (crawler.success_count + crawler.error_count) > 0 else 0),
                'avg_time_per_page': (crawler.total_crawl_time / crawler.success_count 
                                     if crawler.success_count > 0 else 0),
                'pages_per_minute': (crawler.success_count / ((time.time() - self.start_time) / 60)
                                    if time.time() > self.start_time else 0),
                'uptime_minutes': (time.time() - self.start_time) / 60,
                'cache_hit_rate': crawler.cache_hit_rate if hasattr(crawler, 'cache_hit_rate') else 0
            },
            'system': {
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(interval=0.1),
                'threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            },
            'content': {
                'total_content_chars': crawler.total_content_chars,
                'avg_content_length': (crawler.total_content_chars / crawler.success_count 
                                      if crawler.success_count > 0 else 0),
                'unique_emails': len(crawler.emails),
                'unique_phones': len(crawler.phones),
                'high_importance_pages': sum(1 for url in crawler.processed_urls_info.values() 
                                            if url.get('importance_score', 0) > 5),
                'sections_crawled': len(crawler.sections_tracked) if hasattr(crawler, 'sections_tracked') else 0
            }
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def check_health(self, crawler) -> Dict[str, Any]:
        """Check crawler health and generate alerts"""
        metrics = self.update_metrics(crawler)
        perf = metrics['performance']
        sys = metrics['system']
        
        new_alerts = []
        
        # Check error rate
        if perf['error_rate'] > self.alert_thresholds.get('error_rate', 0.1):
            new_alerts.append(f"High error rate: {perf['error_rate']:.2%}")
        
        # Check memory usage
        if sys['memory_usage_mb'] > self.alert_thresholds.get('memory_mb', 1024):
            new_alerts.append(f"High memory usage: {sys['memory_usage_mb']:.1f}MB")
        
        # Check response time
        if perf['avg_time_per_page'] > self.alert_thresholds.get('avg_response_time', 10.0):
            new_alerts.append(f"Slow response time: {perf['avg_time_per_page']:.1f}s")
        
        # Check success rate
        if perf['success_rate'] < self.alert_thresholds.get('success_rate', 0.8):
            new_alerts.append(f"Low success rate: {perf['success_rate']:.2%}")
        
        # Add new alerts
        self.alerts.extend(new_alerts)
        
        health_status = {
            'status': 'healthy' if not new_alerts else 'warning' if len(new_alerts) < 3 else 'critical',
            'alerts': new_alerts,
            'total_alerts': len(self.alerts),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Trigger health check event
        if hasattr(crawler, 'events'):
            crawler.events.trigger('health_check', health_status)
        
        return health_status
    
    def generate_report(self, format: str = 'text') -> str:
        """Generate health report"""
        if not self.metrics_history:
            return "No data available"
        
        latest = self.metrics_history[-1]
        
        if format == 'json':
            return json.dumps({
                'health': latest,
                'alerts_history': list(self.alerts[-10:])
            }, indent=2)
        
        # Text format
        report = f"""
        ╔═══════════════════════════════════════════════════╗
        ║            CRAWLER HEALTH REPORT                  ║
        ╚═══════════════════════════════════════════════════╝
        
        📊 Performance Metrics:
        ------------------------
        • URLs Visited: {latest['performance']['urls_visited']:,}
        • Queue Size: {latest['performance']['queue_size']:,}
        • Success Rate: {latest['performance']['success_rate']:.2%}
        • Error Rate: {latest['performance']['error_rate']:.2%}
        • Avg Time/Page: {latest['performance']['avg_time_per_page']:.2f}s
        • Pages/Minute: {latest['performance']['pages_per_minute']:.1f}
        • Uptime: {latest['performance']['uptime_minutes']:.1f} minutes
        • Cache Hit Rate: {latest['performance'].get('cache_hit_rate', 0):.1%}
        
        💻 System Resources:
        --------------------
        • Memory Usage: {latest['system']['memory_usage_mb']:.1f} MB
        • CPU Usage: {latest['system']['cpu_percent']:.1f}%
        • Threads: {latest['system']['threads']}
        • Open Files: {latest['system']['open_files']}
        
        📝 Content Statistics:
        ----------------------
        • Total Characters: {latest['content']['total_content_chars']:,}
        • Avg Content Length: {latest['content']['avg_content_length']:.0f}
        • Unique Emails: {latest['content']['unique_emails']}
        • Unique Phones: {latest['content']['unique_phones']}
        • High Importance Pages: {latest['content']['high_importance_pages']}
        • Sections Crawled: {latest['content']['sections_crawled']}
        
        ⚠️  Active Alerts: {len(self.alerts[-5:])}
        --------------------
        {chr(10).join(f'  • {alert}' for alert in self.alerts[-5:]) if self.alerts[-5:] else '  None'}
        
        ════════════════════════════════════════════════════
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return report
    
    def reset(self):
        """Reset monitor"""
        self.metrics_history.clear()
        self.alerts.clear()
        self.start_time = time.time()
        self.last_health_check = time.time()


class EnhancedWebCrawler:
    """
    Advanced Web Crawler with Complete Configuration Integration
    =============================================================
    """
    
    def __init__(self, 
                 base_url: str = "https://www.arbin.com/",
                 max_pages: int = None,
                 max_depth: int = None,
                 request_delay: float = None,
                 timeout: int = None,
                 enable_async: bool = None,
                 enable_cache: bool = None,
                 cache_db_path: str = None,
                 user_agent: str = None):
        
        # Load configuration
        self.config = CRAWLER_CONFIG if CONFIG_AVAILABLE else {}
        
        # Apply configuration with overrides
        self.base_url = base_url.rstrip('/')
        self.base_domain = urlparse(base_url).netloc
        
        # Limits from config or parameters
        limits = self.config.get('limits', {})
        self.max_pages = max_pages or limits.get('max_pages_per_run', 500)
        self.max_depth = max_depth or limits.get('max_depth', 5)
        
        # Timing from config or parameters
        timing = self.config.get('timing', {})
        self.request_delay = request_delay or timing.get('base_request_delay', 0.5)
        
        # Timeouts from config or parameters
        timeout_config = self.config.get('timeout', {})
        self.timeout = timeout or timeout_config.get('connect', 15)
        
        # Features from config or parameters
        performance = self.config.get('performance', {})
        self.enable_async = enable_async if enable_async is not None else performance.get('enable_async', True)
        self.max_workers = performance.get('max_workers', 5)
        
        # Cache from config or parameters
        cache_config = self.config.get('cache', {})
        self.enable_cache = enable_cache if enable_cache is not None else cache_config.get('enabled', True)
        
        # Content processing from config
        content_config = self.config.get('content', {})
        self.enable_encoding_detection = content_config.get('encoding_detection', True)
        self.enable_language_detection = content_config.get('language_detection', True)
        self.deduplication_threshold = content_config.get('deduplication_threshold', 0.8)
        self.enable_importance_scoring = content_config.get('importance_scoring', True)
        self.extract_emails = content_config.get('extract_emails', True)
        self.extract_phones = content_config.get('extract_phones', True)
        
        # Load patterns from config
        self.priority_patterns = URL_PATTERNS.get('priority_patterns', []) if CONFIG_AVAILABLE else []
        self.exclude_patterns = URL_PATTERNS.get('exclude_patterns', []) if CONFIG_AVAILABLE else []
        self.section_patterns = URL_PATTERNS.get('sections', {}) if CONFIG_AVAILABLE else {}
        
        # Load other config values
        self.important_keywords = TECH_KEYWORDS if CONFIG_AVAILABLE else []
        self.user_agents = USER_AGENTS if CONFIG_AVAILABLE else []
        self.recrawl_intervals = RECRAWL_INTERVALS if CONFIG_AVAILABLE else {}
        self.anti_bot_patterns = ANTI_BOT_PATTERNS if CONFIG_AVAILABLE else []
        self.selenium_config = SELENIUM_CONFIG if CONFIG_AVAILABLE else {}
        
        # State tracking
        self.visited_urls: Set[str] = set()
        self.urls_to_visit = deque()
        self.processed_urls_info: Dict[str, Dict] = {}

        # Progress tracking variables
        self.discovered_urls: Set[str] = set()
        self.crawled_urls: Set[str] = set()
        self.failed_urls: Dict[str, Any] = {}
        self.skipped_urls: Dict[str, str] = {}
        self.sitemap_urls: Set[str] = set()
        self.non_sitemap_urls: Set[str] = set()
        
        self.start_time = None
        self.total_urls_to_crawl = 0
        self.running = False
        
        # Priority tracking
        self.priority_distribution = {
            "high": 0,      # priority > 3.0
            "medium": 0,    # priority 1.5-3.0
            "low": 0        # priority < 1.5
        }
        
        # Section tracking
        self.sections_tracked = defaultdict(set)
        
        # Content tracking
        self.emails: Set[str] = set()
        self.phones: Set[str] = set()
        self.total_content_chars = 0
        self.success_count = 0
        self.error_count = 0
        self.total_crawl_time = 0.0
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_hit_rate = 0.0
        
        # Performance tracking
        self.start_time = time.time()
        
        # Initialize config-driven components
        self.cache = ConfigDrivenCrawlCache() if self.enable_cache else None
        self.monitor = ConfigDrivenCrawlerMonitor()
        self.events = CrawlerEvents()
        self.rate_limiter = ConfigDrivenRateLimiter()
        
        # HTTP session setup
        self.session = requests.Session()
        default_agent = user_agent or (random.choice(self.user_agents) if self.user_agents else 
                                      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        self.session.headers.update({
            'User-Agent': default_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1'
        })
        
        # Robots.txt parser
        self.robot_parser = RobotFileParser()
        self.robot_parser.set_url(urljoin(self.base_url, "robots.txt"))
        try:
            self.robot_parser.read()
            logger.info(f"Loaded robots.txt from {self.base_url}")
        except Exception as e:
            logger.warning(f"Could not read robots.txt: {e}")
        
        # Load previous crawl data
        self.load_previous_crawls()
        
        logger.info(f"Initialized EnhancedWebCrawler for {self.base_url} with config integration")
        logger.info(f"Configuration: max_pages={self.max_pages}, max_depth={self.max_depth}, "
                   f"async={self.enable_async}, cache={self.enable_cache}")
    
    def load_previous_crawls(self):
        """Load previously crawled data from cache"""
        if self.cache and hasattr(self.cache, 'get_stats'):
            stats = self.cache.get_stats()
            if stats.get('enabled', False):
                logger.info(f"Cache loaded: {stats['total_entries']} entries, {stats['database_size_mb']:.2f} MB")
    
    # ===========================================================================
    # Configuration-driven methods
    # ===========================================================================
    
    def _apply_config_timing(self):
        """Apply timing configuration"""
        if CONFIG_AVAILABLE:
            timing = self.config.get('timing', {})
            self.request_delay = timing.get('base_request_delay', 0.5)
    
    def _get_retry_config(self) -> Dict:
        """Get retry configuration"""
        return self.config.get('retry', {}) if CONFIG_AVAILABLE else {}
    
    def _get_anti_bot_config(self) -> Dict:
        """Get anti-bot configuration"""
        return self.config.get('anti_bot', {}) if CONFIG_AVAILABLE else {}
    
    def _get_skip_config(self) -> Dict:
        """Get skip configuration"""
        return self.config.get('skip_after', {}) if CONFIG_AVAILABLE else {}
    
    def _should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped based on config"""
        skip_config = self._get_skip_config()
        
        # Check if URL has failed too many times
        if url in self.failed_urls:
            fail_info = self.failed_urls[url]
            attempts = fail_info.get('attempts', 0) if isinstance(fail_info, dict) else 1
            
            max_consecutive = skip_config.get('consecutive_errors', 5)
            if attempts >= max_consecutive:
                return True
        
        # Check total errors
        max_total_errors = skip_config.get('total_errors', 20)
        if self.error_count >= max_total_errors:
            logger.warning(f"Skipping due to total errors ({self.error_count} >= {max_total_errors})")
            return True
        
        # Check anti-bot detections
        max_anti_bot = skip_config.get('anti_bot_detections', 3)
        if hasattr(self.rate_limiter, 'anti_bot_detections'):
            if self.rate_limiter.anti_bot_detections >= max_anti_bot:
                logger.warning(f"Skipping due to anti-bot detections ({self.rate_limiter.anti_bot_detections} >= {max_anti_bot})")
                return True
        
        return False
    
    # ===========================================================================
    # URL Processing Utilities (updated with config)
    # ===========================================================================
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL: remove fragments, trailing slashes, etc."""
        if not url:
            return ""
        
        # Remove fragment
        url = urldefrag(url)[0]
        
        # Parse URL
        parsed = urlparse(url)
        
        # Normalize path
        path = parsed.path
        if path.endswith('/') and len(path) > 1:
            path = path.rstrip('/')
        
        # Reconstruct URL
        normalized = parsed._replace(path=path).geturl()
        
        # Ensure HTTPS for base domain
        if parsed.netloc == self.base_domain and parsed.scheme == 'http':
            normalized = normalized.replace('http://', 'https://', 1)
        
        return normalized
    
    def check_robots_permission(self, url: str) -> bool:
        """Check if crawling is allowed by robots.txt"""
        try:
            return self.robot_parser.can_fetch("*", url)
        except:
            return True  # Default to allow if robots.txt can't be read
    
    def should_crawl_url(self, url: str, parent_url: str = None) -> bool:
        """Determine if URL should be crawled"""
        url = self.normalize_url(url)
        
        # Basic validation
        if not url:
            return False
        
        # Check domain
        parsed = urlparse(url)
        if parsed.netloc and parsed.netloc != self.base_domain:
            return False
        
        # Check robots.txt
        if not self.check_robots_permission(url):
            logger.debug(f"Robots.txt disallows: {url}")
            return False
        
        # Check exclusion patterns from config
        url_lower = url.lower()
        for pattern in self.exclude_patterns:
            if re.search(pattern, url_lower, re.IGNORECASE):
                logger.debug(f"URL excluded by pattern {pattern}: {url}")
                return False
        
        # Check file extensions
        excluded_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        if any(url_lower.endswith(ext) for ext in excluded_extensions):
            logger.debug(f"URL excluded (document file): {url}")
            return False
        
        return True
    
    def get_url_priority(self, url: str) -> float:
        """Calculate priority score for URL (higher = more important)"""
        priority = 1.0
        url_lower = url.lower()
        
        # Check priority patterns from config
        for pattern, weight, _ in self.priority_patterns:
            if re.search(pattern, url_lower):
                priority *= weight
        
        # Penalize already visited URLs
        if url in self.visited_urls:
            priority *= 0.5
        
        # Boost homepage
        if url == self.base_url or url == f"{self.base_url}/":
            priority *= 2.0
        
        # Penalize based on depth (estimated from URL structure)
        parsed = urlparse(url)
        depth = len([p for p in parsed.path.split('/') if p])
        if depth > 3:
            priority *= max(0.5, 1.0 - (depth - 3) * 0.1)
        
        return round(priority, 2)
    
    # ===========================================================================
    # Content Analysis and Processing (updated with config)
    # ===========================================================================
    
    def detect_language_and_encoding(self, content: bytes) -> Tuple[str, str]:
        """Detect language and encoding of content"""
        if not self.enable_encoding_detection and not self.enable_language_detection:
            return 'en', 'utf-8'
        
        # Detect encoding
        encoding = 'utf-8'
        if self.enable_encoding_detection:
            encoding_result = chardet.detect(content)
            encoding = encoding_result['encoding'] or 'utf-8'
        
        # Detect language
        language = 'en'
        if self.enable_language_detection:
            try:
                sample = content[:5000].decode(encoding, errors='ignore')
                try:
                    language = detect(sample)
                except LangDetectException:
                    language = 'en'
            except:
                language = 'en'
        
        return language, encoding
    
    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from page"""
        metadata = {
            'title': '',
            'description': '',
            'keywords': [],
            'headers': {},
            'links_count': 0,
            'images_count': 0,
            'meta_tags': {}
        }
        
        # Title
        if soup.title and soup.title.string:
            metadata['title'] = soup.title.string.strip()
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower() or meta.get('property', '').lower()
            content = meta.get('content', '')
            if name and content:
                metadata['meta_tags'][name] = content
        
        # Specific meta tags
        if 'description' in metadata['meta_tags']:
            metadata['description'] = metadata['meta_tags']['description']
        if 'keywords' in metadata['meta_tags']:
            metadata['keywords'] = [
                k.strip() for k in metadata['meta_tags']['keywords'].split(',')
            ]
        
        # Headers
        for i in range(1, 7):
            headers = soup.find_all(f'h{i}')
            if headers:
                metadata['headers'][f'h{i}'] = [
                    h.get_text(strip=True) for h in headers[:10]
                ]
        
        # Counts
        metadata['links_count'] = len(soup.find_all('a', href=True))
        metadata['images_count'] = len(soup.find_all('img'))
        
        # OpenGraph / Twitter Cards
        og_tags = {}
        for meta in soup.find_all('meta', property=re.compile(r'^og:')):
            og_tags[meta['property']] = meta.get('content', '')
        if og_tags:
            metadata['og_tags'] = og_tags
        
        return metadata
    
    def extract_priority_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract main content with priority selectors"""
        content_data = {}
        
        # Content selectors ordered by priority
        content_selectors = [
            ('main', 'main'),
            ('article', 'article'),
            ('[role="main"]', 'role_main'),
            ('.main-content', 'main_content'),
            ('#main-content', 'id_main_content'),
            ('.content', 'content'),
            ('#content', 'id_content'),
            ('[class*="content"]', 'class_content'),
            ('[id*="content"]', 'id_contains_content'),
            ('.post-content', 'post_content'),
            ('.entry-content', 'entry_content'),
            ('.article-content', 'article_content'),
            ('section', 'section'),
            ('.body', 'body'),
            ('.text', 'text'),
            ('.description', 'description')
        ]
        
        for selector, name in content_selectors:
            try:
                elements = soup.select(selector)
                for i, elem in enumerate(elements[:3]):  # Limit to first 3 matches
                    text = elem.get_text(separator=' ', strip=True)
                    if text and len(text) > 100:
                        content_data[f"{name}_{i}"] = {
                            'text': text,
                            'selector': selector,
                            'length': len(text),
                            'html_tag': elem.name
                        }
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
        
        # Fallback: get all text
        if not content_data:
            all_text = soup.get_text(separator=' ', strip=True)
            if all_text and len(all_text) > 50:
                content_data['full_page'] = {
                    'text': all_text,
                    'selector': 'full_page',
                    'length': len(all_text),
                    'html_tag': 'document'
                }
        
        return content_data
    
    def analyze_content_importance(self, text: str, url: str, metadata: Dict) -> float:
        """Calculate importance score for content"""
        if not self.enable_importance_scoring:
            return 1.0
        
        importance_score = 0.0
        text_lower = text.lower()
        url_lower = url.lower()
        
        # Keyword scoring from config
        for keyword in self.important_keywords:
            keyword_lower = keyword.lower()
            # Count occurrences
            count = text_lower.count(keyword_lower)
            if count > 0:
                importance_score += min(count * 0.5, 3.0)  # Cap at 3 per keyword
        
        # URL pattern scoring from config
        for pattern, weight, _ in self.priority_patterns:
            if re.search(pattern, url_lower):
                importance_score += weight
        
        # Content length scoring
        if len(text) > 5000:
            importance_score += 3.0
        elif len(text) > 2000:
            importance_score += 2.0
        elif len(text) > 500:
            importance_score += 1.0
        
        # Header scoring
        if 'headers' in metadata:
            for header_list in metadata['headers'].values():
                for header in header_list:
                    header_lower = header.lower()
                    for keyword in self.important_keywords:
                        if keyword.lower() in header_lower:
                            importance_score += 1.0
        
        # Title scoring
        title = metadata.get('title', '').lower()
        for keyword in self.important_keywords:
            if keyword.lower() in title:
                importance_score += 2.0
        
        # Normalize score
        importance_score = min(importance_score, 20.0)  # Cap at 20
        
        return round(importance_score, 2)
    
    def is_near_duplicate(self, text1: str, text2: str) -> bool:
        """Detect near-duplicate content using fuzzy matching"""
        threshold = self.deduplication_threshold
        
        # Normalize texts
        def normalize(t):
            # Remove extra whitespace and convert to lowercase
            t = re.sub(r'\s+', ' ', t.lower().strip())
            # Remove common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = [w for w in t.split() if w not in common_words and len(w) > 2]
            return ' '.join(words[:100])  # Use first 100 words
        
        norm1 = normalize(text1[:2000])  # Use first 2000 chars
        norm2 = normalize(text2[:2000])
        
        if not norm1 or not norm2:
            return False
        
        # Quick hash check
        if hash(norm1) == hash(norm2):
            return True
        
        # Fuzzy matching
        ratio = SequenceMatcher(None, norm1, norm2).ratio()
        return ratio > threshold
    
    # ===========================================================================
    # Network and Request Handling (updated with config)
    # ===========================================================================
    
    def resolve_redirects(self, url: str, max_redirects: int = 5) -> Tuple[str, List[str]]:
        """Follow redirect chain and return final URL"""
        redirect_chain = []
        current_url = url
        
        for _ in range(max_redirects):
            try:
                response = self.session.head(
                    current_url,
                    timeout=self.timeout,
                    allow_redirects=False
                )
                
                if response.status_code in (301, 302, 303, 307, 308):
                    redirect_url = response.headers.get('Location', '')
                    if not redirect_url:
                        break
                    
                    redirect_url = urljoin(current_url, redirect_url)
                    redirect_url = self.normalize_url(redirect_url)
                    
                    redirect_chain.append(redirect_url)
                    current_url = redirect_url
                else:
                    break
            except Exception as e:
                logger.debug(f"Redirect resolution failed: {e}")
                break
        
        return current_url, redirect_chain
    
    def detect_anti_bot(self, response: requests.Response, html: str = None) -> bool:
        """Detect anti-bot measures"""
        if html is None:
            html = response.text
        
        html_lower = html.lower()
        
        # Check response headers
        server = response.headers.get('Server', '').lower()
        for indicator in ['cloudflare', 'incapsula', 'akamai', 'imperva']:
            if indicator in server:
                return True
        
        # Check status code
        if response.status_code in [403, 429, 503]:
            return True
        
        # Check HTML content for anti-bot patterns from config
        for pattern in self.anti_bot_patterns:
            if pattern in html_lower:
                return True
        
        # Check for CAPTCHA forms
        if ('captcha' in html_lower or 'recaptcha' in html_lower or 
            'g-recaptcha' in html_lower or 'h-captcha' in html_lower):
            return True
        
        return False
    
    def handle_anti_bot(self, url: str) -> bool:
        """Handle anti-bot detection with config-driven strategies"""
        logger.warning(f"Anti-bot detected for {url}")
        
        anti_bot_config = self._get_anti_bot_config()
        max_attempts = anti_bot_config.get('max_attempts', 3)
        
        # Track anti-bot attempts
        if not hasattr(self, '_anti_bot_count'):
            self._anti_bot_count = 0
        self._anti_bot_count += 1
        
        # Record in rate limiter
        if hasattr(self.rate_limiter, 'record_anti_bot'):
            self.rate_limiter.record_anti_bot()
        
        # Trigger anti-bot event
        self.events.trigger('anti_bot_detected', {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'attempt': self._anti_bot_count,
            'max_attempts': max_attempts
        })
        
        # Check if should skip based on config
        skip_config = self._get_skip_config()
        max_anti_bot = skip_config.get('anti_bot_detections', 3)
        if self._anti_bot_count > max_anti_bot:
            logger.error(f"Too many anti-bot attempts for {url}, skipping...")
            self._mark_url_as_skipped(url, "too_many_anti_bot_attempts")
            return False
        
        # Apply strategies based on attempt count
        strategies = [
            self._mild_anti_bot_response,
            self._moderate_anti_bot_response,
            self._aggressive_anti_bot_response
        ]
        
        # Apply strategy based on attempt count
        strategy_idx = min(self._anti_bot_count - 1, len(strategies) - 1)
        strategy = strategies[strategy_idx]
        
        try:
            logger.info(f"Trying anti-bot strategy #{self._anti_bot_count}: {strategy.__name__}")
            if strategy():
                logger.info(f"Anti-bot strategy successful for {url}")
                return True
        except Exception as e:
            logger.error(f"Anti-bot strategy {strategy.__name__} failed: {e}")
        
        return False

    def _mild_anti_bot_response(self) -> bool:
        """Mild anti-bot response: short delay + user agent rotation"""
        # Rotate user agent
        if self.user_agents:
            new_agent = random.choice(self.user_agents)
            self.session.headers.update({'User-Agent': new_agent})
        
        # Short random delay (2-5 seconds)
        delay = random.uniform(2, 5)
        logger.info(f"Mild anti-bot response: Rotated UA, waiting {delay:.1f}s")
        time.sleep(delay)
        
        return True

    def _moderate_anti_bot_response(self) -> bool:
        """Moderate anti-bot response: medium delay + header changes"""
        # Change multiple headers
        headers_to_update = {
            'User-Agent': random.choice(self.user_agents) if self.user_agents else self.session.headers['User-Agent'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1' if random.random() > 0.5 else '0',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '0' if random.random() > 0.5 else '1'
        }
        self.session.headers.update(headers_to_update)
        
        # Medium delay (5-15 seconds)
        delay = random.uniform(5, 15)
        logger.info(f"Moderate anti-bot response: Updated headers, waiting {delay:.1f}s")
        time.sleep(delay)
        
        return True

    def _aggressive_anti_bot_response(self) -> bool:
        """Aggressive anti-bot response: longer delay + session reset"""
        # Reset session
        self.session.close()
        self.session = requests.Session()
        
        # Set completely new headers
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents) if self.user_agents else 'Mozilla/5.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': random.choice(['en-US,en;q=0.9', 'en-GB,en;q=0.8', 'en;q=0.7']),
            'Accept-Encoding': random.choice(['gzip, deflate, br', 'gzip, deflate']),
            'DNT': str(random.randint(0, 1)),
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': random.choice(['max-age=0', 'no-cache']),
            'Pragma': random.choice(['no-cache', ''])
        })
        
        # Add referer if available
        if hasattr(self, 'last_successful_url'):
            self.session.headers.update({'Referer': self.last_successful_url})
        
        # Longer delay but capped
        delay = min(30, random.uniform(10, 30))
        logger.info(f"Aggressive anti-bot response: Reset session, waiting {delay:.1f}s")
        time.sleep(delay)
        
        return True
    
    # ===========================================================================
    # Crawling Logic (updated with config)
    # ===========================================================================
    
    def crawl_with_selenium(self, url: str) -> Optional[BeautifulSoup]:
        """Crawl JavaScript-heavy pages with Selenium using config"""
        if not SELENIUM_AVAILABLE:
            return None
        
        try:
            options = Options()
            
            # Apply Selenium config
            if self.selenium_config.get('headless', True):
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            
            window_size = self.selenium_config.get('window_size', '1920,1080')
            options.add_argument(f'--window-size={window_size}')
            
            user_agent = self.selenium_config.get('user_agent') or random.choice(self.user_agents)
            options.add_argument(f'user-agent={user_agent}')
            
            driver = webdriver.Chrome(options=options)
            
            # Apply timeouts from config
            page_load_timeout = self.selenium_config.get('page_load_timeout', 30)
            script_timeout = self.selenium_config.get('script_timeout', 30)
            driver.set_page_load_timeout(page_load_timeout)
            driver.set_script_timeout(script_timeout)
            
            driver.get(url)
            
            # Wait for page load with configurable delay
            selenium_delay = self.config.get('timing', {}).get('selenium_delay', 3.0)
            time.sleep(selenium_delay)
            
            # Try to wait for dynamic content
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except:
                pass
            
            # Get rendered HTML
            html = driver.page_source
            driver.quit()
            
            return BeautifulSoup(html, 'html.parser')
            
        except Exception as e:
            logger.error(f"Selenium crawl failed for {url}: {e}")
            return None
    
    def crawl_page(self, url: str, depth: int = 0, force_recrawl: bool = False, 
               parent_url: str = None, link_text: str = "") -> Dict[str, Any]:
        """
        Crawl a single page with config-driven retry logic
        """
        start_time = time.time()
        url = self.normalize_url(url)
        
        # Track URL discovery
        self.discovered_urls.add(url)
        self._track_url_by_section(url)
        
        # Check if should skip based on config
        if self._should_skip_url(url):
            logger.debug(f"Skipping {url} - failed too many times previously")
            return {}
        
        # Apply rate limiting
        self.rate_limiter.wait()
        
        # Track current crawl
        logger.info(f"🔍 Crawling: {url} (depth: {depth})")
        
        # Resolve redirects
        try:
            final_url, redirect_chain = self.resolve_redirects(url)
            if final_url != url:
                logger.info(f"Redirect: {url} → {final_url}")
        except Exception as e:
            logger.warning(f"Redirect resolution failed for {url}: {e}")
            final_url = url
            redirect_chain = []
        
        # Check cache first (unless force recrawl)
        if not force_recrawl and self.cache:
            cached_data = self.cache.get(final_url)
            if cached_data and not self.cache.should_recrawl(final_url, cached_data):
                self.cache_hits += 1
                logger.info(f"Cache hit for {final_url}")
                
                # Trigger cache hit event
                self.events.trigger('cache_hit', {
                    'url': final_url,
                    'cached_at': cached_data['crawled_at'],
                    'importance_score': cached_data['importance_score']
                })
                
                # Update cache statistics
                self._update_cache_stats()
                
                return {
                    'url': final_url,
                    'original_url': url,
                    'redirect_chain': redirect_chain,
                    'title': cached_data['title'],
                    'content': '',  # Not loading from cache to save memory
                    'source': 'cache',
                    'source_type': 'cache',
                    'depth': depth,
                    'crawled_at': cached_data['crawled_at'],
                    'content_length': cached_data['content_length'],
                    'importance_score': cached_data['importance_score'],
                    'status_code': cached_data.get('status_code', 200),
                    'metadata': json.loads(cached_data['metadata']) if cached_data['metadata'] else {},
                    'content_hash': cached_data['content_hash'],
                    'is_updated': False,
                    'previous_hash': cached_data['content_hash'],
                    'parent_url': parent_url,
                    'link_text': link_text[:100] if link_text else '',
                    'language': cached_data['language'],
                    'encoding': cached_data['encoding'],
                    'use_selenium': False,
                    'response_time': 0,
                    'attempts_used': 0,
                    'cached': True
                }
        
        self.cache_misses += 1
        self._update_cache_stats()
        
        # Attempt to fetch page with config-driven retry logic
        retry_config = self._get_retry_config()
        max_attempts = retry_config.get('max_attempts', 2)
        backoff_factor = retry_config.get('backoff_factor', 1.5)
        max_backoff = retry_config.get('max_backoff', 30.0)
        retry_codes = retry_config.get('retry_codes', [408, 429, 500, 502, 503, 504])
        
        response = None
        use_selenium = False
        last_exception = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Calculate timeout for this attempt
                attempt_timeout = min(self.timeout * attempt, 30)
                
                # Rotate headers
                headers = self._get_rotated_headers()
                
                response = self.session.get(
                    final_url,
                    timeout=attempt_timeout,
                    allow_redirects=False,
                    stream=True,
                    headers=headers
                )
                
                # Check for anti-bot
                if self.detect_anti_bot(response):
                    logger.warning(f"Anti-bot detected on attempt {attempt} for {final_url}")
                    
                    if attempt == max_attempts:
                        self._handle_failed_url(url, "anti_bot_detection_final", parent_url, attempt)
                        return {}
                    
                    # Handle anti-bot
                    if not self.handle_anti_bot(final_url):
                        continue
                    
                    # Try again after anti-bot handling
                    continue
                
                # Check status code
                if response.status_code >= 400:
                    if response.status_code in retry_codes and attempt < max_attempts:
                        logger.warning(f"HTTP {response.status_code} on attempt {attempt}, retrying...")
                        
                        # Apply exponential backoff
                        backoff_time = min(backoff_factor ** attempt, max_backoff)
                        time.sleep(backoff_time)
                        continue
                    else:
                        # Don't retry for client errors (4xx) except 429
                        if response.status_code < 500 and response.status_code != 429:
                            self._handle_failed_url(url, f"http_{response.status_code}", parent_url, attempt)
                            return {}
                
                response.raise_for_status()
                break  # Success, exit retry loop
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"Attempt {attempt}/{max_attempts} timeout for {final_url}")
                
                if attempt == max_attempts:
                    self._handle_failed_url(url, f"timeout_after_{attempt}_attempts", parent_url, attempt)
                    
                    # Try Selenium as last resort for important pages
                    if depth < 2 and SELENIUM_AVAILABLE:
                        logger.info(f"Trying Selenium as last resort for {final_url}")
                        soup = self.crawl_with_selenium(final_url)
                        if soup:
                            use_selenium = True
                            # Create mock response for Selenium
                            response = type('obj', (object,), {
                                'status_code': 200,
                                'text': str(soup),
                                'content': str(soup).encode('utf-8'),
                                'headers': {'Content-Type': 'text/html'},
                                'encoding': 'utf-8'
                            })()
                            break
                    return {}
                
                # Apply exponential backoff
                backoff_time = min(backoff_factor ** attempt, max_backoff)
                logger.info(f"Backoff {backoff_time}s before retry...")
                time.sleep(backoff_time)
                
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(f"Attempt {attempt}/{max_attempts} connection error for {final_url}")
                
                if attempt == max_attempts:
                    self._handle_failed_url(url, "connection_error", parent_url, attempt)
                    return {}
                
                # Reset session for connection errors
                self._reset_session()
                time.sleep(3)
                
            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code if hasattr(e, 'response') else 'unknown'
                logger.warning(f"Attempt {attempt}/{max_attempts} HTTP error {status_code} for {final_url}")
                
                # Handle specific status codes
                if status_code in [403, 404, 410]:
                    self._handle_failed_url(url, f"http_{status_code}", parent_url, attempt)
                    return {}
                elif status_code == 429:  # Rate limited
                    if attempt == max_attempts:
                        self._handle_failed_url(url, "rate_limited", parent_url, attempt)
                        return {}
                    
                    # Apply backoff for rate limiting
                    backoff_time = min(30, 5 * attempt)
                    logger.warning(f"Rate limited, backing off for {backoff_time}s")
                    time.sleep(backoff_time)
                else:
                    if attempt == max_attempts:
                        self._handle_failed_url(url, f"http_error_{status_code}", parent_url, attempt)
                        return {}
                
            except requests.RequestException as e:
                last_exception = e
                logger.warning(f"Attempt {attempt}/{max_attempts} failed for {final_url}: {e}")
                
                if attempt == max_attempts:
                    self._handle_failed_url(url, f"request_error: {str(e)[:50]}", parent_url, attempt)
                    return {}
        
        # If we get here without a response, try Selenium as last resort
        if not response and not use_selenium and SELENIUM_AVAILABLE and depth < 2:
            logger.info(f"Trying Selenium as last resort for {final_url}")
            soup = self.crawl_with_selenium(final_url)
            if soup:
                use_selenium = True
                # Create mock response for Selenium
                response = type('obj', (object,), {
                    'status_code': 200,
                    'text': str(soup),
                    'content': str(soup).encode('utf-8'),
                    'headers': {'Content-Type': 'text/html'},
                    'encoding': 'utf-8'
                })()
        
        if not response and not use_selenium:
            logger.error(f"All attempts failed for {final_url}")
            self._handle_failed_url(url, "all_attempts_failed", parent_url, max_attempts)
            return {}
        
        # Process successful response
        try:
            response_time = time.time() - start_time
            
            # Parse HTML
            if use_selenium:
                html = str(soup)
                soup_obj = soup
            else:
                if response.encoding is None:
                    response.encoding = response.apparent_encoding or 'utf-8'
                html = response.text
                soup_obj = BeautifulSoup(html, 'html.parser')
            
            # Validate page
            if not self._is_valid_page(soup_obj, html):
                logger.warning(f"Invalid page detected for {final_url}")
                self._handle_failed_url(url, "invalid_page_content", parent_url, 1)
                return {}
            
            # Remove unwanted elements
            for selector in ['script', 'style', 'nav', 'footer', 'header', 
                            'aside', 'form', 'iframe', 'noscript', 'svg',
                            '.advertisement', '.ad', '.ads', '[class*="ad"]',
                            '.social-share', '.share-buttons', '.comments']:
                for element in soup_obj.select(selector):
                    element.decompose()
            
            # Extract content
            content_data = self.extract_priority_content(soup_obj, final_url)
            metadata = self.extract_metadata(soup_obj, final_url)
            
            # Combine all text
            combined_text = ' '.join([d['text'] for d in content_data.values()])
            cleaned_text = ' '.join(
                chunk.strip() for line in combined_text.splitlines() 
                for chunk in line.split("  ") if chunk.strip()
            )
            
            # Check content length against config limits
            limits = self.config.get('limits', {})
            min_content = limits.get('min_content_length', 100)
            max_content = limits.get('max_content_length', 10485760)
            
            if len(cleaned_text) < min_content and not use_selenium:
                logger.warning(f"Content too short ({len(cleaned_text)} chars) for {final_url}")
                # Still continue but log warning
            
            if len(cleaned_text) > max_content:
                logger.warning(f"Content too long ({len(cleaned_text)} chars), truncating for {final_url}")
                cleaned_text = cleaned_text[:max_content]
            
            # Analyze content
            importance_score = self.analyze_content_importance(cleaned_text, final_url, metadata)
            
            # Extract emails and phones if enabled
            emails = []
            phones = []
            
            if self.extract_emails:
                emails = list(set(re.findall(
                    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", 
                    cleaned_text
                )))
            
            if self.extract_phones:
                phones = list(set(re.findall(
                    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}", 
                    cleaned_text
                )))
            
            # Update collections
            self.emails.update(emails)
            self.phones.update(phones)
            self.total_content_chars += len(cleaned_text)
            
            # Calculate hash
            content_hash = hashlib.md5(cleaned_text.encode('utf-8')).hexdigest()
            
            # Check for updates in cache
            is_updated = False
            previous_hash = None
            
            if self.cache:
                cached = self.cache.get(final_url)
                if cached:
                    previous_hash = cached['content_hash']
                    is_updated = previous_hash != content_hash
            
            # Detect language and encoding
            language, encoding = self.detect_language_and_encoding(
                response.content if response else cleaned_text.encode('utf-8')
            )
            
            # Get URL section
            section = self._get_url_section(final_url)
            
            # Update priority distribution
            self._update_priority_distribution(importance_score)
            
            # Record successful crawl
            self.last_successful_url = final_url
            
            # Prepare result
            result = {
                'url': final_url,
                'original_url': url,
                'redirect_chain': redirect_chain,
                'title': metadata['title'][:200] if metadata['title'] else '',
                'content': cleaned_text,
                'source': 'web',
                'source_type': 'web',
                'depth': depth,
                'crawled_at': datetime.now().isoformat(),
                'content_length': len(cleaned_text),
                'importance_score': round(importance_score, 2),
                'status_code': response.status_code if response else 200,
                'metadata': metadata,
                'content_sections': content_data,
                'content_hash': content_hash,
                'is_updated': is_updated,
                'previous_hash': previous_hash,
                'parent_url': parent_url,
                'link_text': link_text[:100] if link_text else '',
                'emails': emails,
                'phones': phones,
                'language': language,
                'encoding': encoding,
                'section': section,
                'use_selenium': use_selenium,
                'response_time': round(response_time, 2),
                'attempts_used': attempt if 'attempt' in locals() else 1,
                'cached': False
            }
            
            # Extract links for further crawling
            if depth < self.max_depth:
                self.extract_and_queue_links(soup_obj, final_url, depth)
            
            # Update cache
            if self.cache:
                cache_data = {
                    'content_hash': content_hash,
                    'crawled_at': result['crawled_at'],
                    'title': result['title'],
                    'content_length': result['content_length'],
                    'importance_score': importance_score,
                    'metadata': metadata,
                    'language': language,
                    'encoding': encoding,
                    'redirect_url': final_url if final_url != url else '',
                    'section': section,
                    'depth': depth,
                    'status_code': result['status_code'],
                    'last_modified': response.headers.get('Last-Modified', '') if response else '',
                    'etag': response.headers.get('ETag', '') if response else '',
                    'crawl_duration': response_time
                }
                self.cache.set(final_url, cache_data)
            
            # Update success metrics
            self.success_count += 1
            self.total_crawl_time += response_time
            
            # Update rate limiter
            if hasattr(self.rate_limiter, 'record_success'):
                self.rate_limiter.record_success()
            
            # Track successful crawl
            self.crawled_urls.add(final_url)
            
            # Trigger event
            self.events.trigger('page_crawled', {
                'url': final_url,
                'depth': depth,
                'importance_score': importance_score,
                'content_length': len(cleaned_text),
                'response_time': response_time,
                'section': section,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(
                f"✅ Crawled: {final_url[:80]}... - "
                f"{len(cleaned_text):,} chars, "
                f"score: {importance_score:.1f}, "
                f"time: {response_time:.1f}s, "
                f"section: {section}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {final_url}: {str(e)[:100]}")
            self._handle_failed_url(url, f"processing_error: {str(e)[:50]}", parent_url, 1)
            return {}
    
    def _update_cache_stats(self):
        """Update cache hit rate statistics"""
        total = self.cache_hits + self.cache_misses
        if total > 0:
            self.cache_hit_rate = self.cache_hits / total
    
    # ===========================================================================
    # Helper Methods
    # ===========================================================================
    
    def _mark_url_as_skipped(self, url: str, reason: str):
        """Mark URL as skipped with reason"""
        self.skipped_urls[url] = reason
        logger.info(f"Marked {url} as skipped: {reason}")
    
    def _handle_failed_url(self, url: str, reason: str, parent_url: str = None, attempts: int = 1):
        """Handle URL that failed to crawl"""
        if url not in self.failed_urls:
            self.failed_urls[url] = {
                'attempts': attempts,
                'last_error': reason,
                'first_failed': datetime.now().isoformat(),
                'parent_url': parent_url
            }
        else:
            fail_info = self.failed_urls[url]
            if isinstance(fail_info, dict):
                fail_info['attempts'] += attempts
                fail_info['last_error'] = reason
                fail_info['last_failed'] = datetime.now().isoformat()
            else:
                # Convert old format to new format
                self.failed_urls[url] = {
                    'attempts': attempts,
                    'last_error': reason,
                    'first_failed': datetime.now().isoformat(),
                    'parent_url': parent_url
                }
        
        self.error_count += 1
        
        # Record error in rate limiter
        if hasattr(self.rate_limiter, 'record_error'):
            self.rate_limiter.record_error()
        
        logger.debug(f"Marked {url} as failed: {reason} (attempts: {attempts})")
    
    def _get_rotated_headers(self) -> Dict:
        """Get headers with rotation"""
        base_headers = dict(self.session.headers)
        
        # Add random elements
        if random.random() > 0.5:
            base_headers['DNT'] = '1'
        
        if random.random() > 0.5 and hasattr(self, 'last_successful_url'):
            base_headers['Referer'] = self.last_successful_url
        
        # Randomly change Accept-Language
        if random.random() > 0.7:
            languages = ['en-US,en;q=0.9', 'en-GB,en;q=0.8', 'en;q=0.7', 'fr-FR,fr;q=0.9']
            base_headers['Accept-Language'] = random.choice(languages)
        
        return base_headers
    
    def _reset_session(self):
        """Reset HTTP session"""
        self.session.close()
        self.session = requests.Session()
        
        # Set default headers
        default_agent = random.choice(self.user_agents) if self.user_agents else 'Mozilla/5.0'
        self.session.headers.update({
            'User-Agent': default_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
    
    def _is_valid_page(self, soup, html: str) -> bool:
        """Check if page is valid (not error page)"""
        # Check for title or content
        title = soup.title.string if soup.title else ''
        body_text = soup.get_text(strip=True)
        
        if len(body_text) < 50 and len(title) < 5:
            return False
        
        # Check for error indicators
        error_indicators = ['error', 'not found', 'forbidden', 'access denied', '404', '500']
        html_lower = html.lower()
        
        if any(indicator in html_lower for indicator in error_indicators):
            # But check if it's just mentioning errors, not actually an error page
            error_mentions = sum(html_lower.count(indicator) for indicator in error_indicators)
            if error_mentions > 3:  # Too many error mentions
                return False
        
        return True
    
    # ===========================================================================
    # Sitemap Discovery and Processing
    # ===========================================================================
    
    def discover_sitemaps(self) -> List[str]:
        """Discover sitemap files"""
        sitemap_urls = set()
        
        # Common sitemap paths
        common_paths = [
            'sitemap.xml',
            'sitemap_index.xml',
            'sitemap-index.xml',
            'sitemap1.xml',
            'sitemap-news.xml',
            'sitemap-products.xml',
            'sitemap-articles.xml',
            'sitemap-blog.xml',
        ]
        
        # Check common paths
        for path in common_paths:
            sitemap_url = urljoin(self.base_url, path)
            try:
                response = self.session.head(sitemap_url, timeout=5)
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'xml' in content_type or 'sitemap' in content_type:
                        sitemap_urls.add(sitemap_url)
                        logger.debug(f"Found sitemap: {sitemap_url}")
            except:
                continue
        
        # Parse robots.txt for Sitemap directives
        robots_url = urljoin(self.base_url, 'robots.txt')
        try:
            response = self.session.get(robots_url, timeout=5)
            if response.status_code == 200:
                for line in response.text.splitlines():
                    line = line.strip()
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        sitemap_url = urljoin(self.base_url, sitemap_url)
                        sitemap_urls.add(sitemap_url)
                        logger.debug(f"Found sitemap in robots.txt: {sitemap_url}")
        except:
            pass
        
        return list(sitemap_urls)
    
    def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """Parse sitemap XML and extract URLs"""
        urls = []
        
        try:
            response = self.session.get(sitemap_url, timeout=10)
            if response.status_code != 200:
                return urls
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            if 'xml' in content_type or 'sitemap' in content_type:
                try:
                    root = ET.fromstring(response.content)
                    
                    # Namespace handling
                    namespace = '{http://www.sitemaps.org/schemas/sitemap/0.9}'
                    
                    # Find URL locations
                    loc_elements = root.findall(f'.//{namespace}loc') or root.findall('.//loc')
                    
                    for loc in loc_elements:
                        url = loc.text.strip()
                        
                        if self._is_valid_url(url):
                            if self.should_crawl_url(url):
                                urls.append(url)
                                self.sitemap_urls.add(url)
                    
                    # Check for sitemap index
                    sitemap_elements = root.findall(f'.//{namespace}sitemap') or root.findall('.//sitemap')
                    for sitemap in sitemap_elements:
                        sitemap_loc = sitemap.find(f'{namespace}loc') or sitemap.find('loc')
                        if sitemap_loc and sitemap_loc.text:
                            nested_urls = self.parse_sitemap(sitemap_loc.text.strip())
                            urls.extend(nested_urls)
                
                except ET.ParseError:
                    soup = BeautifulSoup(response.content, 'xml')
                    loc_tags = soup.find_all('loc')
                    for tag in loc_tags:
                        url = tag.text.strip()
                        if self._is_valid_url(url) and self.should_crawl_url(url):
                            urls.append(url)
                            self.sitemap_urls.add(url)
            
            elif 'text/plain' in content_type and 'robots.txt' in sitemap_url:
                for line in response.text.splitlines():
                    line = line.strip()
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        nested_urls = self.parse_sitemap(sitemap_url)
                        urls.extend(nested_urls)
            
        except Exception as e:
            logger.error(f"Error parsing sitemap {sitemap_url}: {e}")
        
        logger.info(f"Parsed {len(urls)} valid URLs from sitemap: {sitemap_url}")
        return urls

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        if not url or not isinstance(url, str):
            return False
        
        if len(url) > 500:
            return False
        
        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:/.-_?&=#%")
        invalid_count = sum(1 for char in url if char not in valid_chars and not char.isspace())
        
        if invalid_count > len(url) * 0.3:
            return False
        
        if not url.startswith(('http://', 'https://')):
            return True
        
        return True
    
    # ===========================================================================
    # Link Extraction and Queue Management
    # ===========================================================================
    
    def extract_and_queue_links(self, soup: BeautifulSoup, current_url: str, depth: int):
        """Extract and queue links from page"""
        links_found = 0
        
        for a_tag in soup.find_all('a', href=True):
            try:
                href = urljoin(current_url, a_tag['href'])
                href = self.normalize_url(href)
                
                if not self.should_crawl_url(href, current_url):
                    continue
                
                # Calculate priority
                priority = self.get_url_priority(href)
                
                # Add to queue if not already visited or queued
                if (href not in self.visited_urls and 
                    href not in [url for url, _, _, _, _, _ in self.urls_to_visit]):
                    
                    link_text = a_tag.get_text(strip=True)[:100]
                    self.urls_to_visit.append((
                        href, 
                        depth + 1, 
                        False,  # force_recrawl
                        current_url, 
                        link_text,
                        priority
                    ))
                    
                    self.non_sitemap_urls.add(href)
                    links_found += 1
                    
                    # Trigger event
                    self.events.trigger('url_discovered', {
                        'url': href,
                        'parent_url': current_url,
                        'depth': depth + 1,
                        'link_text': link_text,
                        'priority': priority,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                self.skipped_urls[href] = f"error_processing: {str(e)}"
                logger.debug(f"Error processing link: {e}")
        
        logger.debug(f"Extracted {links_found} links from {current_url}")
    
    # ===========================================================================
    # Progress Tracking Methods
    # ===========================================================================
    
    def _reset_tracking(self):
        """Reset all tracking variables"""
        self.discovered_urls.clear()
        self.crawled_urls.clear()
        self.failed_urls.clear()
        self.skipped_urls.clear()
        self.sitemap_urls.clear()
        self.non_sitemap_urls.clear()
        self.priority_distribution = {"high": 0, "medium": 0, "low": 0}
        self.sections_tracked.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_hit_rate = 0.0
    
    def _track_url_by_section(self, url: str):
        """Track URL by website section"""
        for section_name, pattern in self.section_patterns.items():
            if re.search(pattern, url.lower()):
                self.sections_tracked[section_name].add(url)
                return
        
        self.sections_tracked["other"].add(url)
    
    def _update_priority_distribution(self, priority: float):
        """Update priority distribution counts"""
        if priority > 3.0:
            self.priority_distribution["high"] += 1
        elif priority > 1.5:
            self.priority_distribution["medium"] += 1
        else:
            self.priority_distribution["low"] += 1
    
    def _get_url_section(self, url: str) -> str:
        """Get section of URL"""
        for section_name, pattern in self.section_patterns.items():
            if re.search(pattern, url.lower()):
                return section_name
        
        return "other"
    
    # ===========================================================================
    # Main Crawling Methods
    # ===========================================================================
    
    def get_initial_urls(self, force_recrawl: bool = False) -> List[str]:
        """Get initial URLs for crawling"""
        initial_urls = []
        
        # Try sitemaps first
        sitemaps = self.discover_sitemaps()
        if sitemaps:
            logger.info(f"Found {len(sitemaps)} sitemaps")
            
            for sitemap in sitemaps:
                urls = self.parse_sitemap(sitemap)
                
                if not force_recrawl:
                    # Filter based on recrawl interval
                    filtered_urls = []
                    for url in urls[:200]:
                        if self.cache:
                            cached = self.cache.get(url)
                            if cached:
                                if not self.cache.should_recrawl(url, cached):
                                    continue
                        filtered_urls.append(url)
                    initial_urls.extend(filtered_urls)
                else:
                    initial_urls.extend(urls[:200])
        
        # Fallback to important sections
        if not initial_urls:
            logger.info("No sitemaps found, using important sections")
            
            important_sections = [
                '/', '/products/', '/software/', '/support/', '/resources/', 
                '/applications/', '/about/', '/news/', '/blog/', '/contact/',
                '/downloads/', '/documentation/', '/faq/', '/help/'
            ]
            
            for section in important_sections:
                url = urljoin(self.base_url, section)
                if self.should_crawl_url(url):
                    initial_urls.append(url)
        
        # Prioritize URLs
        prioritized_urls = []
        for url in initial_urls:
            priority = self.get_url_priority(url)
            prioritized_urls.append((url, priority))
        
        # Sort by priority (highest first)
        prioritized_urls.sort(key=lambda x: x[1], reverse=True)
        
        # Return top URLs
        top_urls = [url for url, _ in prioritized_urls[:100]]
        logger.info(f"Selected {len(top_urls)} initial URLs for crawling")
        
        return top_urls
    
    def crawl_site(self, force_recrawl: bool = False, max_workers: int = None) -> List[Dict[str, Any]]:
        """
        Main method to crawl the entire site
        """
        # Reset tracking
        self._reset_tracking()
        self.start_time = time.time()
        self.running = True
        logger.info(f"🚀 Starting {'FORCED ' if force_recrawl else ''}crawl of {self.base_url}")
        
        # Reset state for new crawl
        self.visited_urls.clear()
        self.urls_to_visit.clear()
        self.emails.clear()
        self.phones.clear()
        self.total_content_chars = 0
        self.success_count = 0
        self.error_count = 0
        self.total_crawl_time = 0.0
        self.monitor.reset()
        
        # Get initial URLs
        initial_urls = self.get_initial_urls(force_recrawl)
        
        # Add initial URLs to queue with priorities
        for url in initial_urls:
            priority = self.get_url_priority(url)
            self.urls_to_visit.append((url, 0, force_recrawl, None, "", priority))
        
        # Sort queue by priority
        self.urls_to_visit = deque(
            sorted(self.urls_to_visit, key=lambda x: x[5], reverse=True)
        )
        
        # Use config max_workers if not specified
        if max_workers is None:
            max_workers = self.max_workers
        
        documents = []
        
        # Choose crawling strategy
        if self.enable_async and max_workers > 1:
            logger.info(f"Using async crawling with {max_workers} workers")
            documents = self._crawl_async(max_workers, force_recrawl)
        else:
            logger.info("Using sequential crawling")
            documents = self._crawl_sequential(force_recrawl)
        
        # Save results
        if documents:
            self.save_crawl_results(documents)
        
        # Generate final report
        health = self.monitor.check_health(self)
        report = self.monitor.generate_report()
        
        logger.info("\n" + "="*60)
        logger.info("CRAWLING COMPLETED")
        logger.info("="*60)
        logger.info(report)
        
        # Update cache stats
        self._update_cache_stats()
        
        # Trigger completion event
        self.events.trigger('complete', {
            'total_documents': len(documents),
            'crawled_urls': len(self.crawled_urls),
            'failed_urls': len(self.failed_urls),
            'skipped_urls': len(self.skipped_urls),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hit_rate,
            'health_status': health['status'],
            'total_time': time.time() - self.start_time,
            'timestamp': datetime.now().isoformat()
        })

        self.running = False
        
        # Generate final report
        final_report = self.generate_crawl_report()
        
        # Save report if enabled
        if self.config.get('monitoring', {}).get('save_reports', True):
            self._save_crawl_report(final_report)
        
        return documents
    
    def _crawl_sequential(self, force_recrawl: bool) -> List[Dict[str, Any]]:
        """Sequential crawling implementation"""
        documents = []
        new_count = updated_count = cached_count = error_count = 0
        
        while self.urls_to_visit and len(documents) < self.max_pages:
            # Get next URL (already sorted by priority)
            url, depth, force, parent, link_text, priority = self.urls_to_visit.popleft()
            
            if url in self.visited_urls:
                continue
            
            self.visited_urls.add(url)
            
            try:
                # Crawl page
                doc = self.crawl_page(url, depth, force, parent, link_text)
                
                if not doc:
                    error_count += 1
                    continue
                
                # Classify result
                if doc.get('cached'):
                    cached_count += 1
                elif doc.get('content'):
                    documents.append(doc)
                    if doc.get('is_updated'):
                        updated_count += 1
                    else:
                        new_count += 1
                else:
                    error_count += 1
                
                # Periodic logging and health checks
                total_processed = len(documents) + cached_count + error_count
                if total_processed % 10 == 0 or total_processed == 1:
                    health = self.monitor.check_health(self)
                    
                    logger.info(
                        f"📊 Progress: {total_processed}/{self.max_pages} | "
                        f"New: {new_count} | Updated: {updated_count} | "
                        f"Cached: {cached_count} | Errors: {error_count} | "
                        f"Queue: {len(self.urls_to_visit)} | "
                        f"Cache Hit Rate: {self.cache_hit_rate:.1%} | "
                        f"Health: {health['status']}"
                    )
                
                # Periodic cache cleanup
                if total_processed % 50 == 0 and self.cache:
                    self.cache.cleanup_old()
                
                # Check health thresholds
                if health['status'] == 'critical' and total_processed > 10:
                    logger.error("Critical health status, stopping crawl")
                    break
                
            except Exception as e:
                logger.error(f"Error in crawl loop for {url}: {e}")
                error_count += 1
                self.events.trigger('error', {
                    'url': url,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return documents
    
    def _crawl_async(self, max_workers: int, force_recrawl: bool) -> List[Dict[str, Any]]:
        """Asynchronous crawling implementation"""
        documents = []
        lock = threading.Lock()
        
        def worker(worker_id: int):
            """Worker function for thread pool"""
            nonlocal documents
            
            while self.running and len(documents) < self.max_pages:
                try:
                    # Get next URL
                    with lock:
                        if not self.urls_to_visit or len(documents) >= self.max_pages:
                            break
                        url, depth, force, parent, link_text, priority = self.urls_to_visit.popleft()
                        
                        if url in self.visited_urls:
                            continue
                        
                        self.visited_urls.add(url)
                    
                    # Crawl page
                    doc = self.crawl_page(url, depth, force, parent, link_text)
                    
                    if doc and doc.get('content') and not doc.get('cached'):
                        with lock:
                            documents.append(doc)
                    
                    # Periodic logging
                    with lock:
                        if len(documents) % 20 == 0:
                            health = self.monitor.check_health(self)
                            logger.info(
                                f"📊 Worker {worker_id}: {len(documents)}/{self.max_pages} | "
                                f"Queue: {len(self.urls_to_visit)} | "
                                f"Health: {health['status']}"
                            )
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
        
        # Create and start workers
        workers = []
        for i in range(max_workers):
            t = threading.Thread(target=worker, args=(i,), name=f"CrawlerWorker-{i}")
            t.daemon = True
            t.start()
            workers.append(t)
        
        # Monitor progress
        try:
            while self.running and any(t.is_alive() for t in workers) and len(documents) < self.max_pages:
                time.sleep(2)
                
                health = self.monitor.check_health(self)
                
                if len(documents) % 50 == 0:
                    logger.info(
                        f"📊 Async Progress: {len(documents)}/{self.max_pages} | "
                        f"Queue: {len(self.urls_to_visit)} | "
                        f"Workers: {sum(1 for t in workers if t.is_alive())}/{max_workers} | "
                        f"Health: {health['status']}"
                    )
                
                # Stop if critical health
                if health['status'] == 'critical' and len(documents) > 10:
                    logger.error("Critical health status, stopping crawl")
                    self.running = False
                    break
        
        except KeyboardInterrupt:
            logger.info("Crawl interrupted by user")
            self.running = False
        
        finally:
            # Wait for workers to finish
            for t in workers:
                t.join(timeout=5)
            
            logger.info(f"Async crawl completed with {len(documents)} documents")
        
        return documents
    
    # ===========================================================================
    # Results and Reporting
    # ===========================================================================
    
    def save_crawl_results(self, documents: List[Dict[str, Any]]):
        """Save crawl results and generate statistics"""
        try:
            # Filter and categorize documents
            new_docs = [d for d in documents if not d.get('is_updated') and not d.get('cached')]
            updated_docs = [d for d in documents if d.get('is_updated')]
            cached_docs = [d for d in documents if d.get('cached')]
            
            # Generate statistics
            stats = {
                'crawl_summary': {
                    'total_pages_crawled': len(documents),
                    'new_documents': len(new_docs),
                    'updated_documents': len(updated_docs),
                    'cached_documents': len(cached_docs),
                    'total_content_chars': self.total_content_chars,
                    'avg_content_length': self.total_content_chars / len(new_docs + updated_docs) if (new_docs + updated_docs) else 0,
                    'unique_emails': len(self.emails),
                    'unique_phones': len(self.phones),
                    'max_depth': max((d.get('depth', 0) for d in documents), default=0),
                    'avg_importance_score': sum((d.get('importance_score', 0) for d in documents)) / len(documents) if documents else 0,
                    'cache_hit_rate': self.cache_hit_rate
                },
                'performance_metrics': {
                    'success_rate': self.success_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 0,
                    'avg_response_time': self.total_crawl_time / self.success_count if self.success_count > 0 else 0,
                    'total_crawl_time': self.total_crawl_time,
                    'pages_per_minute': self.success_count / ((time.time() - self.start_time) / 60) if time.time() > self.start_time else 0
                },
                'url_distribution': {
                    'by_depth': defaultdict(int),
                    'by_importance': {
                        'high': sum(1 for d in documents if d.get('importance_score', 0) > 5),
                        'medium': sum(1 for d in documents if 2 < d.get('importance_score', 0) <= 5),
                        'low': sum(1 for d in documents if d.get('importance_score', 0) <= 2)
                    },
                    'by_section': defaultdict(int)
                },
                'top_documents': []
            }
            
            # Depth and section distribution
            for doc in documents:
                depth = doc.get('depth', 0)
                stats['url_distribution']['by_depth'][depth] += 1
                
                section = doc.get('section', 'unknown')
                stats['url_distribution']['by_section'][section] += 1
            
            # Top documents by importance
            top_docs = sorted([d for d in documents if not d.get('cached')], 
                             key=lambda x: x.get('importance_score', 0), reverse=True)[:10]
            stats['top_documents'] = [
                {
                    'url': doc['url'],
                    'title': doc.get('title', ''),
                    'importance_score': doc.get('importance_score', 0),
                    'content_length': doc.get('content_length', 0),
                    'depth': doc.get('depth', 0),
                    'section': doc.get('section', 'unknown')
                }
                for doc in top_docs
            ]
            
            # Sample content preview
            if new_docs or updated_docs:
                sample = (new_docs + updated_docs)[0]
                stats['sample_document'] = {
                    'url': sample['url'],
                    'title': sample.get('title', '')[:100],
                    'content_preview': sample.get('content', '')[:300] + '...' if len(sample.get('content', '')) > 300 else sample.get('content', ''),
                    'importance_score': sample.get('importance_score', 0),
                    'language': sample.get('language', 'unknown'),
                    'section': sample.get('section', 'unknown')
                }
            
            # Export statistics
            logger.info(f"\n📊 CRAWLING STATISTICS:")
            logger.info(f"   Total pages: {stats['crawl_summary']['total_pages_crawled']}")
            logger.info(f"   New documents: {stats['crawl_summary']['new_documents']}")
            logger.info(f"   Updated documents: {stats['crawl_summary']['updated_documents']}")
            logger.info(f"   Cached documents: {stats['crawl_summary']['cached_documents']}")
            logger.info(f"   Cache hit rate: {stats['crawl_summary']['cache_hit_rate']:.1%}")
            logger.info(f"   Unique emails: {stats['crawl_summary']['unique_emails']}")
            logger.info(f"   Unique phones: {stats['crawl_summary']['unique_phones']}")
            logger.info(f"   Success rate: {stats['performance_metrics']['success_rate']:.2%}")
            logger.info(f"   Avg response time: {stats['performance_metrics']['avg_response_time']:.2f}s")
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = f"crawl_stats_{timestamp}.json"
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Statistics saved to: {stats_file}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error saving crawl results: {e}")
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current crawler status"""
        health = self.monitor.check_health(self)
        
        return {
            'status': 'running' if self.running else 'idle',
            'running': self.running,
            'health': health,
            'progress': {
                'visited': len(self.visited_urls),
                'queued': len(self.urls_to_visit),
                'success': self.success_count,
                'errors': self.error_count,
                'max_pages': self.max_pages,
                'crawled_urls': len(self.crawled_urls),
                'failed_urls': len(self.failed_urls),
                'skipped_urls': len(self.skipped_urls)
            },
            'content': {
                'total_chars': self.total_content_chars,
                'unique_emails': len(self.emails),
                'unique_phones': len(self.phones)
            },
            'performance': {
                'avg_time_per_page': self.total_crawl_time / self.success_count if self.success_count > 0 else 0,
                'uptime_minutes': (time.time() - self.start_time) / 60,
                'cache_hit_rate': self.cache_hit_rate
            },
            'cache': self.cache.get_stats() if self.cache else None,
            'timestamp': datetime.now().isoformat()
        }
    
    # ===========================================================================
    # Reporting Methods
    # ===========================================================================
    
    def get_crawl_progress(self) -> Dict[str, Any]:
        """Get detailed crawl progress statistics"""
        total_discovered = len(self.discovered_urls)
        total_crawled = len(self.crawled_urls)
        total_failed = len(self.failed_urls)
        total_skipped = len(self.skipped_urls)
        
        # Calculate percentages
        if total_discovered > 0:
            crawled_percent = (total_crawled / total_discovered) * 100
            failed_percent = (total_failed / total_discovered) * 100
            skipped_percent = (total_skipped / total_discovered) * 100
            pending_percent = 100 - crawled_percent - failed_percent - skipped_percent
        else:
            crawled_percent = failed_percent = skipped_percent = pending_percent = 0
        
        # Get pending URLs
        pending_urls = list(self.discovered_urls - self.crawled_urls - 
                           set(self.failed_urls.keys()) - set(self.skipped_urls.keys()))
        
        # Get important pending URLs
        important_pending = []
        for url in pending_urls:
            priority = self.get_url_priority(url)
            if priority > 2.0:
                important_pending.append({
                    "url": url,
                    "priority": priority,
                    "reason": "high_priority_not_crawled"
                })
        
        # Sort by priority
        important_pending.sort(key=lambda x: x["priority"], reverse=True)
        
        # Section coverage
        section_coverage = {}
        for section, urls in self.sections_tracked.items():
            crawled_in_section = len(urls & self.crawled_urls)
            total_in_section = len(urls)
            if total_in_section > 0:
                coverage = (crawled_in_section / total_in_section) * 100
                section_coverage[section] = {
                    "crawled": crawled_in_section,
                    "total": total_in_section,
                    "coverage_percent": round(coverage, 2)
                }
        
        return {
            "summary": {
                "total_discovered": total_discovered,
                "total_crawled": total_crawled,
                "total_failed": total_failed,
                "total_skipped": total_skipped,
                "total_pending": len(pending_urls),
                "crawled_percent": round(crawled_percent, 2),
                "failed_percent": round(failed_percent, 2),
                "skipped_percent": round(skipped_percent, 2),
                "pending_percent": round(pending_percent, 2),
                "success_rate": round((total_crawled / total_discovered * 100), 2) if total_discovered > 0 else 0,
                "cache_hit_rate": self.cache_hit_rate
            },
            "source_breakdown": {
                "from_sitemap": len(self.sitemap_urls),
                "from_link_discovery": len(self.non_sitemap_urls),
                "sitemap_coverage": round((len(self.sitemap_urls & self.crawled_urls) / len(self.sitemap_urls) * 100), 2) if self.sitemap_urls else 0
            },
            "priority_distribution": self.priority_distribution,
            "section_coverage": section_coverage,
            "pending_urls": {
                "total": len(pending_urls),
                "important": important_pending[:20],
                "sample": pending_urls[:50]
            },
            "failed_urls_sample": dict(list(self.failed_urls.items())[:20]),
            "skipped_urls_sample": dict(list(self.skipped_urls.items())[:20]),
            "crawl_duration": time.time() - self.start_time if self.start_time else 0
        }
    
    def generate_crawl_report(self) -> Dict[str, Any]:
        """Generate comprehensive crawl report"""
        from datetime import datetime, timedelta
        
        progress = self.get_crawl_progress()
        
        report = {
            "metadata": {
                "base_url": self.base_url,
                "crawl_start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "report_time": datetime.now().isoformat(),
                "crawl_duration_seconds": progress.get("crawl_duration", 0),
                "crawl_duration_human": str(timedelta(seconds=progress.get("crawl_duration", 0))) if progress.get("crawl_duration") else "N/A",
                "max_pages": self.max_pages,
                "max_depth": self.max_depth,
                "config_used": "crawler_config.py" if CONFIG_AVAILABLE else "default",
                "cache_enabled": self.enable_cache
            },
            "progress_summary": progress["summary"],
            "coverage_analysis": {
                "sitemap_coverage_percent": progress["source_breakdown"]["sitemap_coverage"],
                "important_pages_crawled": self.priority_distribution["high"],
                "total_important_pages": sum(1 for url in self.discovered_urls 
                                            if self.get_url_priority(url) > 3.0),
                "section_coverage": progress["section_coverage"],
                "cache_performance": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": self.cache_hit_rate
                }
            },
            "uncovered_areas": self.get_uncovered_areas(),
            "pending_urls": progress["pending_urls"],
            "failed_urls_summary": {
                "total": len(self.failed_urls),
                "sample": progress["failed_urls_sample"]
            },
            "recommendations": self._generate_recommendations(progress),
            "next_steps": self._generate_next_steps(progress),
            "configuration_summary": {
                "limits": {
                    "max_pages": self.max_pages,
                    "max_depth": self.max_depth,
                    "max_workers": self.max_workers
                },
                "features": {
                    "async_enabled": self.enable_async,
                    "cache_enabled": self.enable_cache,
                    "selenium_enabled": SELENIUM_AVAILABLE and self.selenium_config.get('enabled', True)
                },
                "timing": {
                    "base_delay": self.request_delay,
                    "cache_hit_rate": self.cache_hit_rate
                }
            }
        }
        
        return report
    
    def get_uncovered_areas(self) -> Dict[str, Any]:
        """Identify uncovered areas of the website"""
        from collections import defaultdict
        
        uncovered = {
            "by_section": defaultdict(list),
            "by_priority_pattern": defaultdict(list),
            "estimated_missing": 0
        }
        
        # Check important sections
        for section_name, pattern in self.section_patterns.items():
            # Check if main section page was crawled
            section_urls = [url for url in self.discovered_urls if re.search(pattern, url.lower())]
            uncrawled_section_urls = [url for url in section_urls if url not in self.crawled_urls]
            
            if uncrawled_section_urls:
                uncovered["by_section"][section_name].extend([
                    {
                        "url": url,
                        "reason": "not_crawled",
                        "priority": self.get_url_priority(url)
                    }
                    for url in uncrawled_section_urls[:5]
                ])
        
        # Check priority patterns
        for pattern, weight, name in self.priority_patterns:
            pattern_urls = [url for url in self.discovered_urls if re.search(pattern, url.lower())]
            uncrawled_pattern_urls = [url for url in pattern_urls if url not in self.crawled_urls]
            
            if uncrawled_pattern_urls:
                uncovered["by_priority_pattern"][name].extend(
                    uncrawled_pattern_urls[:10]
                )
        
        # Estimate missing URLs
        total_crawled = len(self.crawled_urls)
        if total_crawled > 0 and self.sitemap_urls:
            sitemap_coverage = len(self.sitemap_urls & self.crawled_urls) / len(self.sitemap_urls)
            uncovered["estimated_missing"] = int((1 - sitemap_coverage) * 100)
        
        return uncovered
    
    def _generate_recommendations(self, progress: Dict) -> List[str]:
        """Generate recommendations based on crawl progress"""
        recommendations = []
        
        crawled_percent = progress["summary"]["crawled_percent"]
        success_rate = progress["summary"]["success_rate"]
        cache_hit_rate = progress["summary"].get("cache_hit_rate", 0)
        
        if crawled_percent < 50:
            recommendations.append("⚠️ Low crawl coverage (<50%). Consider increasing max_pages or max_depth.")
        
        if success_rate < 80:
            recommendations.append(f"⚠️ Success rate is {success_rate:.1f}%. Check for access issues or anti-bot measures.")
        
        if cache_hit_rate > 0.5:
            recommendations.append(f"✅ Good cache hit rate: {cache_hit_rate:.1%}. Cache is effectively reducing redundant crawling.")
        elif cache_hit_rate < 0.2 and self.enable_cache:
            recommendations.append(f"⚠️ Low cache hit rate: {cache_hit_rate:.1%}. Consider adjusting recrawl intervals.")
        
        if progress["summary"]["failed_percent"] > 20:
            recommendations.append("🚨 High failure rate (>20%). Check network/access issues or server blocks.")
        
        # Check sitemap coverage
        sitemap_coverage = progress["source_breakdown"]["sitemap_coverage"]
        if sitemap_coverage < 80 and self.sitemap_urls:
            recommendations.append(f"📊 Sitemap coverage is {sitemap_coverage:.1f}%. Consider prioritizing sitemap URLs.")
        
        return recommendations
    
    def _generate_next_steps(self, progress: Dict) -> List[Dict]:
        """Generate next steps for continuing the crawl"""
        next_steps = []
        
        # Continue crawling pending URLs
        pending_count = progress["summary"]["total_pending"]
        if pending_count > 0:
            next_steps.append({
                "action": "continue_crawl",
                "priority": "high",
                "description": f"Crawl {pending_count} pending URLs",
                "estimated_time_minutes": int(pending_count * 2),
                "urls_count": pending_count
            })
        
        # Retry failed URLs
        failed_count = len(self.failed_urls)
        if failed_count > 0:
            next_steps.append({
                "action": "retry_failed",
                "priority": "medium",
                "description": f"Retry {failed_count} failed URLs",
                "urls": list(self.failed_urls.keys())[:10],
                "suggested_changes": [
                    "Increase timeouts",
                    "Add longer delays",
                    "Try Selenium for JavaScript pages"
                ]
            })
        
        # Focus on uncovered sections
        uncovered = self.get_uncovered_areas()
        for section, urls in uncovered["by_section"].items():
            if urls:
                next_steps.append({
                    "action": "crawl_section",
                    "priority": "high",
                    "section": section,
                    "description": f"Crawl uncovered section: {section}",
                    "urls": [u["url"] for u in urls[:5]],
                    "total_urls": len(urls)
                })
        
        return next_steps
    
    def _save_crawl_report(self, report: Dict):
        """Save crawl report to file"""
        import json
        from pathlib import Path
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path("reports")
            report_dir.mkdir(exist_ok=True)
            
            report_file = report_dir / f"crawl_report_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Crawl report saved to: {report_file}")
            
        except Exception as e:
            logger.error(f"Error saving crawl report: {e}")
    
    # ===========================================================================
    # Utility Methods
    # ===========================================================================
    
    def export_results(self, format: str = 'json', filename: str = None) -> str:
        """Export crawl results in specified format"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crawl_results_{timestamp}.{format}"
        
        # Get current documents from cache
        documents = []
        if self.cache:
            with self.cache.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM crawl_cache ORDER BY importance_score DESC"
                )
                for row in cursor:
                    doc = dict(row)
                    doc['metadata'] = json.loads(doc['metadata'])
                    documents.append(doc)
        
        if format == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            import csv
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['URL', 'Title', 'Importance', 'Length', 'Language', 'Section', 'Crawled At'])
                for doc in documents:
                    writer.writerow([
                        doc['url'],
                        doc['title'][:100],
                        doc['importance_score'],
                        doc['content_length'],
                        doc['language'],
                        doc.get('section', 'unknown'),
                        doc['crawled_at']
                    ])
        
        logger.info(f"Exported {len(documents)} documents to {filename}")
        return filename
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up crawler resources...")
        
        self.running = False
        
        # Close session
        if hasattr(self, 'session'):
            self.session.close()
        
        # Cleanup cache
        if self.cache:
            self.cache.cleanup_old()
        
        logger.info("Cleanup completed")
    
    def get_unreachable_urls_report(self) -> Dict[str, Any]:
        """Generate report of URLs that couldn't be crawled"""
        from collections import defaultdict
        
        report = {
            'summary': {
                'total_failed_urls': len(self.failed_urls),
                'total_skipped_urls': len(self.skipped_urls),
                'total_unreachable': len(self.failed_urls) + len(self.skipped_urls),
                'success_rate': (len(self.crawled_urls) / len(self.discovered_urls) * 100 
                            if self.discovered_urls else 0),
                'crawl_success_count': len(self.crawled_urls),
                'crawl_failure_count': len(self.failed_urls),
                'skipped_count': len(self.skipped_urls)
            },
            'failed_urls_details': {},
            'skipped_urls_details': {},
            'error_analysis': {
                'by_error_type': defaultdict(list),
                'by_section': defaultdict(list),
                'by_depth': defaultdict(list),
                'most_common_errors': []
            },
            'recommendations': [],
            'retry_suggestions': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Process failed URLs
        for url, info in self.failed_urls.items():
            if isinstance(info, dict):
                error = info.get('last_error', 'unknown')
                attempts = info.get('attempts', 1)
                parent = info.get('parent_url')
            else:
                error = str(info)
                attempts = 1
                parent = None
            
            error_type = self._classify_error_type(error)
            
            report['failed_urls_details'][url] = {
                'error': error[:200],
                'error_type': error_type,
                'attempts': attempts,
                'parent_url': parent,
                'depth': self._estimate_url_depth(url),
                'section': self._get_url_section(url),
                'can_retry': attempts < 3 and error_type not in ['permanent', 'not_found']
            }
            
            # Add to error analysis
            report['error_analysis']['by_error_type'][error_type].append(url)
            report['error_analysis']['by_section'][self._get_url_section(url)].append(url)
            report['error_analysis']['by_depth'][self._estimate_url_depth(url)].append(url)
        
        # Process skipped URLs
        for url, reason in self.skipped_urls.items():
            report['skipped_urls_details'][url] = {
                'reason': reason,
                'section': self._get_url_section(url),
                'depth': self._estimate_url_depth(url),
                'skip_type': self._classify_skip_reason(reason)
            }
        
        # Analyze error patterns
        error_counts = defaultdict(int)
        for url_info in report['failed_urls_details'].values():
            error_type = url_info['error_type']
            error_counts[error_type] += 1
        
        # Get most common errors
        report['error_analysis']['most_common_errors'] = sorted(
            error_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Generate recommendations
        self._generate_unreachable_recommendations(report)
        
        # Generate retry suggestions
        self._generate_unreachable_retry_suggestions(report)
        
        return report
    
    def _classify_error_type(self, error_msg: str) -> str:
        """Classify error type"""
        error_lower = error_msg.lower()
        
        if 'timeout' in error_lower:
            return 'timeout'
        elif 'connection' in error_lower:
            return 'connection'
        elif any(code in error_lower for code in ['403', 'forbidden', 'access denied']):
            return 'permission'
        elif any(code in error_lower for code in ['404', 'not found']):
            return 'not_found'
        elif any(code in error_lower for code in ['429', 'rate limit', 'too many requests']):
            return 'rate_limit'
        elif any(keyword in error_lower for keyword in ['captcha', 'cloudflare', 'bot detected', 'anti-bot']):
            return 'anti_bot'
        elif 'ssl' in error_lower:
            return 'ssl'
        elif 'dns' in error_lower:
            return 'dns'
        elif 'selenium' in error_lower:
            return 'selenium_failed'
        else:
            return 'unknown'
    
    def _classify_skip_reason(self, reason: str) -> str:
        """Classify skip reason"""
        reason_lower = reason.lower()
        
        if 'pattern' in reason_lower:
            return 'excluded_by_pattern'
        elif 'extension' in reason_lower:
            return 'excluded_extension'
        elif 'external' in reason_lower:
            return 'external_domain'
        elif 'robots' in reason_lower:
            return 'robots_txt'
        elif 'domain' in reason_lower:
            return 'wrong_domain'
        else:
            return 'other'
    
    def _estimate_url_depth(self, url: str) -> int:
        """Estimate URL depth"""
        if url == self.base_url or url == f"{self.base_url}/":
            return 0
        
        parsed = urlparse(url)
        path = parsed.path
        
        if not path or path == '/':
            return 0
        
        segments = [s for s in path.split('/') if s]
        return min(len(segments), 5)
    
    def _generate_unreachable_recommendations(self, report: Dict):
        """Generate recommendations for unreachable URLs"""
        error_counts = report['error_analysis']['most_common_errors']
        
        for error_type, count in error_counts:
            if error_type == 'timeout' and count > 5:
                report['recommendations'].append(
                    f"⚠️ {count} timeout errors. Consider increasing timeout settings or using proxy servers."
                )
            elif error_type == 'connection' and count > 3:
                report['recommendations'].append(
                    f"⚠️ {count} connection errors. Check network connectivity or try again later."
                )
            elif error_type == 'anti_bot' and count > 2:
                report['recommendations'].append(
                    f"⚠️ {count} anti-bot detections. Consider implementing proxy rotation or increasing delays."
                )
            elif error_type == 'rate_limit' and count > 1:
                report['recommendations'].append(
                    f"⚠️ {count} rate limit errors. Reduce crawl speed or implement exponential backoff."
                )
        
        if report['summary']['success_rate'] < 70:
            report['recommendations'].append(
                f"⚠️ Low success rate ({report['summary']['success_rate']:.1f}%). "
                "Review failed URLs and adjust crawler configuration."
            )
        
        if len(report['failed_urls_details']) > 20:
            report['recommendations'].append(
                f"⚠️ High number of failed URLs ({len(report['failed_urls_details'])}). "
                "Consider running in smaller batches or with longer delays."
            )
    
    def _generate_unreachable_retry_suggestions(self, report: Dict):
        """Generate retry suggestions for unreachable URLs"""
        retry_candidates = []
        
        for url, info in report['failed_urls_details'].items():
            if info.get('can_retry', False):
                retry_candidates.append({
                    'url': url,
                    'error': info['error'],
                    'attempts': info['attempts'],
                    'priority': 'high' if info['section'] in ['products', 'software'] else 'medium'
                })
        
        if retry_candidates:
            report['retry_suggestions'] = {
                'total_retry_candidates': len(retry_candidates),
                'high_priority': [c for c in retry_candidates if c['priority'] == 'high'],
                'medium_priority': [c for c in retry_candidates if c['priority'] == 'medium'],
                'suggested_approach': 'Retry with increased delays and rotated user agents'
            }
    
    def __del__(self):
        """Destructor"""
        self.cleanup()


# Backward compatibility alias
WebCrawler = EnhancedWebCrawler


def create_crawler(base_url: str = "https://www.arbin.com/", **kwargs) -> EnhancedWebCrawler:
    """Factory function to create crawler with configuration"""
    return EnhancedWebCrawler(base_url=base_url, **kwargs)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create crawler
    crawler = EnhancedWebCrawler(
        base_url="https://www.arbin.com/",
        max_pages=100,
        max_depth=3,
        enable_async=True,
        enable_cache=True
    )
    
    # Register event handlers
    def on_page_crawled(data):
        logger.debug(f"Page crawled: {data['url']} (score: {data['importance_score']})")
    
    def on_error(data):
        logger.error(f"Crawl error: {data['url']} - {data['error']}")
    
    def on_cache_hit(data):
        logger.info(f"Cache hit: {data['url']} (cached: {data['cached_at']})")
    
    crawler.events.register_handler('page_crawled', on_page_crawled)
    crawler.events.register_handler('error', on_error)
    crawler.events.register_handler('cache_hit', on_cache_hit)
    
    # Start crawling
    try:
        results = crawler.crawl_site(force_recrawl=False, max_workers=3)
        print(f"\n✅ Crawling completed! Retrieved {len(results)} documents.")
        
        # Export results
        crawler.export_results('json')
        
        # Generate reports
        report = crawler.generate_crawl_report()
        print(f"\n📋 Generated crawl report with {len(report.get('recommendations', []))} recommendations")
        
        # Show unreachable URLs report
        unreachable_report = crawler.get_unreachable_urls_report()
        print(f"\n⚠️  Unreachable URLs: {unreachable_report['summary']['total_unreachable']}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Crawling interrupted by user")
    except Exception as e:
        print(f"\n❌ Crawling failed: {e}")
    finally:
        crawler.cleanup()