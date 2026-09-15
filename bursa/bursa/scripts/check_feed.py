"""Verify the quote feed on THIS machine. Run before trusting real-time.

    python -m scripts.check_feed
    python -m scripts.check_feed --ticker 0270 --force-moomoo

`core.feed.MoomooFeed` was written from moomoo's published API docs and has
never run against a live OpenD gateway -- there is no OpenD and no moomoo
account in the environment it was written in. This script is the missing
verification step, and it has to run where OpenD is.

It checks, in order:

1. Is anything listening on OpenD's port?
2. Is moomoo's Python SDK installed?
3. Does a quote come back, and does it look like a real Bursa price?
4. How does it compare to Yahoo's delayed quote for the same name?

Step 4 is the one that catches a wrong symbol convention. If moomoo says
RM 1.71 and Yahoo says RM 1.71, the `MY.<code>` mapping is right. If moomoo
returns a price from a different listing -- or a suspiciously round number --
the codes are being resolved to the wrong instrument, and every downstream
number would be wrong in a way nothing else would catch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.feed import (
    MoomooFeed, YahooFeed, get_feed, opend_listening, to_moomoo, to_yahoo,
)
from core.quotes import market_status

OK, BAD, WARN = "  [ok]  ", "  [FAIL]", "  [warn]"

# Two quotes for the same stock at the same moment should agree closely. A
# gap wider than this is not delay -- it is a different instrument.
AGREEMENT_TOL = 0.05


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="0270")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11111)
    ap.add_argument("--force-moomoo", action="store_true",
                    help="fail instead of falling back to Yahoo")
    args = ap.parse_args()

    print(f"Bursa is currently: {market_status().value}")
    print(f"checking ticker {args.ticker}  "
          f"(yahoo {to_yahoo(args.ticker)} / moomoo {to_moomoo(args.ticker)})\n")

    # 1. gateway
    listening = opend_listening(args.host, args.port)
    print(f"{OK if listening else WARN} OpenD on {args.host}:{args.port}: "
          f"{'listening' if listening else 'not listening'}")
    if not listening:
        print("         Start OpenD and log in. Until then everything falls "
              "back to Yahoo's\n         delayed feed, which is fine for "
              "tracking but not for pricing an order.")

    # 2. SDK
    sdk = None
    for name in ("moomoo", "futu"):
        try:
            __import__(name)
            sdk = name
            break
        except ImportError:
            continue
    print(f"{OK if sdk else WARN} moomoo SDK: "
          f"{sdk or 'not installed (pip install moomoo-api)'}")

    # 3. real-time quote
    mq = None
    if listening and sdk:
        feed = MoomooFeed(host=args.host, port=args.port)
        mq = feed.quote(args.ticker)
        feed.close()
        if mq.ok:
            print(f"{OK} moomoo quote: RM {mq.price:.3f} at {mq.quoted_at}")
        else:
            print(f"{BAD} moomoo quote failed: {mq.error}")

    # 4. cross-check
    yq = YahooFeed().quote(args.ticker)
    if yq.ok:
        print(f"{OK} yahoo quote:  RM {yq.price:.3f} at {yq.quoted_at} "
              f"({yq.staleness()})")
    else:
        print(f"{BAD} yahoo quote failed: {yq.error}")

    if mq is not None and mq.ok and yq.ok:
        gap = abs(mq.price - yq.price) / yq.price
        if gap <= AGREEMENT_TOL:
            print(f"{OK} the two agree within {gap:.2%} -- the MY.<code> "
                  f"mapping resolves to the right instrument")
        else:
            print(f"{BAD} they differ by {gap:.1%}. That is too wide to be "
                  f"delay.\n         Check the symbol convention BEFORE "
                  f"trading off this feed -- a wrong\n         mapping gives "
                  f"you a confident price for the wrong company.")
            return 1

    active = get_feed(host=args.host, port=args.port)
    print(f"\n  active feed: {active.name}")
    print(f"  {active.describe()}")
    print(f"  real-time: {active.realtime}")

    if args.force_moomoo and not active.realtime:
        print(f"\n{BAD} --force-moomoo set but the feed fell back to Yahoo.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
