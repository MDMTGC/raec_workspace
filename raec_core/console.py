"""
Cross-platform console output utilities.

Provides safe print functions that work on Windows (cp1252)
as well as Unix terminals with full Unicode support.
"""

import sys

# Detect if stdout can handle Unicode safely
def _can_unicode() -> bool:
    """Check if the current stdout supports Unicode output."""
    try:
        encoding = getattr(sys.stdout, 'encoding', None) or 'ascii'
        return encoding.lower() in ('utf-8', 'utf8', 'utf-16', 'utf16')
    except Exception:
        return False

UNICODE_SAFE = _can_unicode()

# ASCII fallbacks for common status indicators
SYMBOLS = {
    # Status marks
    'check': '[OK]' if not UNICODE_SAFE else '\u2713',      # ✓
    'cross': '[X]' if not UNICODE_SAFE else '\u2717',       # ✗
    'ok': '[OK]' if not UNICODE_SAFE else '\u2705',         # ✅
    'fail': '[FAIL]' if not UNICODE_SAFE else '\u274c',     # ❌
    'warn': '[!]' if not UNICODE_SAFE else '\u26a0',        # ⚠
    'info': '[i]' if not UNICODE_SAFE else '\u2139',        # ℹ

    # Progress/state
    'gear': '[*]' if not UNICODE_SAFE else '\u2699',        # ⚙
    'play': '[>]' if not UNICODE_SAFE else '\u25b6',        # ▶
    'pause': '[||]' if not UNICODE_SAFE else '\u23f8',      # ⏸
    'stop': '[.]' if not UNICODE_SAFE else '\u23f9',        # ⏹

    # Objects
    'brain': '[R]' if not UNICODE_SAFE else '\U0001f9e0',   # 🧠
    'clipboard': '[=]' if not UNICODE_SAFE else '\U0001f4cb', # 📋
    'search': '[?]' if not UNICODE_SAFE else '\U0001f50d',  # 🔍
    'bulb': '[!]' if not UNICODE_SAFE else '\U0001f4a1',    # 💡
    'chart': '[#]' if not UNICODE_SAFE else '\U0001f4ca',   # 📊
    'target': '[o]' if not UNICODE_SAFE else '\U0001f3af', # 🎯
    'tools': '[T]' if not UNICODE_SAFE else '\U0001f527',   # 🔧
    'robot': '[R]' if not UNICODE_SAFE else '\U0001f916',   # 🤖
    'new': '[+]' if not UNICODE_SAFE else '\U0001f195',     # 🆕
    'handshake': '[&]' if not UNICODE_SAFE else '\U0001f91d', # 🤝
}


def sym(name: str) -> str:
    """Return a console-safe symbol by name."""
    return SYMBOLS.get(name, f'[{name}]')


def safe_print(*args, **kwargs) -> None:
    """
    Print that gracefully degrades on encoding errors.

    Falls back to replacing unrepresentable characters with '?'.
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        message = ' '.join(str(a) for a in args)
        encoded = message.encode(sys.stdout.encoding or 'ascii', errors='replace')
        print(encoded.decode(sys.stdout.encoding or 'ascii', errors='replace'), **kwargs)


# Convenience shortcuts
def log_ok(msg: str) -> None:
    safe_print(f"   {sym('check')} {msg}")

def log_fail(msg: str) -> None:
    safe_print(f"   {sym('cross')} {msg}")

def log_warn(msg: str) -> None:
    safe_print(f"   {sym('warn')} {msg}")

def log_info(msg: str) -> None:
    safe_print(f"   {sym('info')} {msg}")

def log_step(msg: str) -> None:
    safe_print(f"{sym('gear')}  {msg}")
