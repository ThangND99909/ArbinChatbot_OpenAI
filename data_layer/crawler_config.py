"""
Crawler Configuration - Complete Safe Settings
"""

CRAWLER_CONFIG = {
    # Timing settings (SAFE limits)
    'timing': {
        'base_request_delay': 0.5,    # Base delay between requests
        'min_delay': 0.3,             # Minimum delay between requests
        'max_delay': 5.0,             # Maximum delay (capped!)
        'jitter_factor': 0.2,         # Random jitter factor (0.2 = ±20%)
        'selenium_delay': 3.0,        # Additional delay for Selenium
    },
    
    # Retry settings
    'retry': {
        'max_attempts': 2,           # Max retry attempts per URL
        'backoff_factor': 1.5,       # Exponential backoff factor
        'max_backoff': 30.0,         # Maximum backoff time
        'retry_codes': [408, 429, 500, 502, 503, 504],  # Status codes to retry
    },
    
    # Anti-bot handling
    'anti_bot': {
        'max_attempts': 3,           # Max anti-bot attempts
        'delay_strategy': 'linear',  # 'linear' or 'exponential'
        'max_delay': 30.0,           # Maximum anti-bot delay
        'strategies': ['mild', 'moderate', 'aggressive'],  # Available strategies
    },
    
    # Timeouts
    'timeout': {
        'connect': 10,    # Connection timeout
        'read': 30,       # Read timeout
        'total': 60,      # Total request timeout
        'selenium': 30,   # Selenium timeout
    },
    
    # Skip patterns
    'skip_after': {
        'consecutive_errors': 5,     # Skip after X consecutive errors
        'total_errors': 20,          # Skip after X total errors
        'anti_bot_detections': 3,    # Skip after X anti-bot detections
        'timeout_attempts': 3,       # Skip after X timeout attempts
    },
    
    # Limits
    'limits': {
        'max_pages_per_run': 500,    # Maximum pages to crawl per run
        'max_depth': 5,              # Maximum crawl depth
        'max_queue_size': 10000,     # Maximum URL queue size
        'max_content_length': 10485760,  # 10MB max content length
        'min_content_length': 100,   # Minimum content length to save
    },
    
    # Content processing
    'content': {
        'encoding_detection': True,
        'language_detection': True,
        'deduplication_threshold': 0.8,  # Similarity threshold for dedup
        'importance_scoring': True,
        'extract_emails': True,
        'extract_phones': True,
    },
    
    # Cache settings
    'cache': {
        'enabled': True,
        'db_path': 'crawler_cache.db',
        'cleanup_days': 90,          # Cleanup cache older than X days
        'max_cache_size_mb': 1024,   # Maximum cache size (1GB)
    },
    
    # Performance
    'performance': {
        'enable_async': True,
        'max_workers': 5,            # Max concurrent workers
        'batch_size': 50,            # Batch size for async processing
        'memory_limit_mb': 1024,     # Memory usage limit
    },
    
    # Monitoring
    'monitoring': {
        'health_check_interval': 60,  # Health check every X seconds
        'alert_thresholds': {
            'error_rate': 0.1,       # Alert if error rate > 10%
            'memory_mb': 1024,       # Alert if memory > 1GB
            'avg_response_time': 10.0,  # Alert if avg response > 10s
            'success_rate': 0.8,     # Alert if success rate < 80%
        },
        'generate_reports': True,
        'save_reports': True,
    }
}

# URL Patterns for intelligent crawling
URL_PATTERNS = {
    # Priority patterns (higher = more important)
    'priority_patterns': [
        (r'/products?/', 3.0, 'products'),
        (r'/software?/', 3.0, 'software'),
        (r'/support?/', 2.5, 'support'),
        (r'/technical[-_]?support/', 2.5, 'support'),
        (r'/resources?/', 2.0, 'resources'),
        (r'/applications?/', 2.0, 'applications'),
        (r'/downloads?/', 2.0, 'downloads'),
        (r'/documentation?/', 2.0, 'documentation'),
        (r'/manuals?/', 2.0, 'documentation'),
        (r'/specifications?/', 2.0, 'documentation'),
        (r'/datasheets?/', 2.0, 'documentation'),
        (r'/brochures?/', 1.5, 'resources'),
        (r'/news?/', 1.5, 'news'),
        (r'/blog?/', 1.5, 'blog'),
        (r'/articles?/', 1.5, 'articles'),
        (r'/tutorials?/', 1.5, 'tutorials'),
        (r'/guides?/', 1.5, 'guides'),
        (r'/faqs?/', 1.5, 'faq'),
        (r'/help/', 1.5, 'help'),
        (r'/knowledge[-_]?base/', 1.5, 'knowledge_base'),
        (r'/about/', 1.0, 'about'),
        (r'/contact/', 1.0, 'contact'),
        (r'/company/', 1.0, 'company'),
        (r'/careers?/', 1.0, 'careers'),
    ],
    
    # Exclusion patterns (won't crawl these)
    'exclude_patterns': [
        r'\.(css|js|jpg|jpeg|png|gif|bmp|svg|ico|webp|mp4|mp3|avi|mov|pdf|zip|rar|tar|gz|exe|dmg|msi)$',
        r'\/cart\/', r'\/checkout\/', r'\/account\/', r'\/login\/', r'\/register\/',
        r'\/wp-admin\/', r'\/wp-content\/', r'\/wp-json\/', r'\/wp-includes\/',
        r'\/cgi-bin\/', r'\.php$', r'\.asp$', r'\.aspx$', r'\.jsp$',
        r'#', r'javascript:', r'mailto:', r'tel:', r'skype:', r'feed:',
        r'\/search\/', r'\/filter\/', r'\/sort\/', r'\?.*sort=', r'\?.*filter=',
        r'\/print\/', r'\/printpreview\/', r'\/pdf\/', r'\/export\/',
        r'\/api\/', r'\/ajax\/', r'\/rest\/', r'\/graphql\/',
        r'\/admin\/', r'\/dashboard\/', r'\/private\/', r'\/secure\/',
        r'\?.*session=', r'\?.*token=', r'\?.*auth=', r'\?.*password=',
    ],
    
    # Section patterns for tracking
    'sections': {
        'products': r'/products?/',
        'software': r'/software?/',
        'support': r'/support?/',
        'resources': r'/resources?/',
        'documentation': r'/doc|manual|guide|spec|datasheet',
        'news': r'/news|blog|articles?|press/',
        'about': r'/about|company|contact|careers?/',
        'home': r'^https?://[^/]+/?$',
    }
}

# Technical keywords for importance scoring (Arbin-specific)
TECH_KEYWORDS = [
    'arbin', 'battery', 'test', 'testing', 'system', 'bt-', 'lbt', 'mbt', 'mits',
    'software', 'pro', 'hardware', 'specification', 'technical', 'data', 'measurement',
    'voltage', 'current', 'capacity', 'cycler', 'tester', 'ev', 'electric', 'vehicle',
    'r&d', 'research', 'development', 'manufacturing', 'quality', 'control',
    'laboratory', 'lab', 'cell', 'lithium', 'ion', 'analysis', 'daq',
    'acquisition', 'calibration', 'accuracy', 'resolution', 'channel', 'module', 'modular',
    'configuration', 'instrument', 'equipment', 'solution', 'technology', 'innovation',
    'energy', 'storage', 'power', 'management', 'safety', 'reliability',
    'performance', 'efficiency', 'monitoring', 'diagnostics', 'protection',
    'charging', 'discharging', 'cycling', 'degradation', 'aging', 'lifetime',
    'temperature', 'thermal', 'impedance', 'resistance', 'conductivity',
    'electrochemical', 'electrode', 'electrolyte', 'anode', 'cathode',
    'soc', 'state of charge', 'soh', 'state of health', 'sop', 'state of power',
    'bms', 'battery management system', 'ev battery', 'stationary storage',
    'grid storage', 'renewable energy', 'sustainability', 'green energy'
]

# User agents for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
]

# Recrawl intervals in days
RECRAWL_INTERVALS = {
    'homepage': 1,
    'product_pages': 7,
    'software_pages': 7,
    'support_pages': 30,
    'documentation': 90,
    'news_blog': 3,
    'resources': 14,
    'applications': 14,
    'downloads': 30,
    'about_contact': 60,
    'default': 14
}

# Anti-bot detection patterns
ANTI_BOT_PATTERNS = [
    'captcha', 'recaptcha', 'hcaptcha', 'access denied', 'blocked',
    'bot detected', 'security check', 'cloudflare', 'incapsula',
    'distil networks', 'imperva', 'akamai', 'barracuda',
    'please verify', 'verification required', 'human verification',
    'security challenge', 'challenge page', 'rate limited',
    'too many requests', 'request blocked', 'suspicious activity'
]

# Selenium configuration
SELENIUM_CONFIG = {
    'enabled': True,
    'headless': True,
    'window_size': '1920,1080',
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'page_load_timeout': 30,
    'script_timeout': 30,
    'implicit_wait': 10,
}

# Export everything
__all__ = [
    'CRAWLER_CONFIG',
    'URL_PATTERNS', 
    'TECH_KEYWORDS',
    'USER_AGENTS',
    'RECRAWL_INTERVALS',
    'ANTI_BOT_PATTERNS',
    'SELENIUM_CONFIG'
]