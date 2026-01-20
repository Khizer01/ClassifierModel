import argparse
import asyncio
import json
import sys

from config import get_settings
from database import DatabaseManager
from model import AdClassifier


async def _classify_keyword(keyword: str, db_manager: DatabaseManager, classifier: AdClassifier) -> int:
    if not keyword or not keyword.strip():
        print("ERROR: keyword is required", file=sys.stderr)
        return 2

    keyword = keyword.strip()

    ads = await db_manager.get_ads_by_keyword(keyword)
    if not ads:
        print(json.dumps({"keyword": keyword, "results": []}, indent=2))
        return 0

    print(f"Processing {len(ads)} ads for keyword: {keyword}")
    results = classifier.classify_batch(keyword, ads)

    output = {
        "keyword": keyword,
        "total_ads": len(ads),
        "results": results,
    }
    print(json.dumps(output, indent=2))
    return 0


async def _interactive_loop(initial_keyword: str | None) -> int:
    settings = get_settings()

    print("Initializing...")
    db_manager = DatabaseManager(settings.database_path)
    await db_manager.ensure_initialized()

    classifier = AdClassifier()
    classifier.load_model()

    if initial_keyword is not None and initial_keyword.strip():
        rc = await _classify_keyword(initial_keyword, db_manager, classifier)
        if rc != 0:
            return rc

    while True:
        try:
            keyword = input("Enter keyword (or 'exit'): ").strip()
        except EOFError:
            return 0

        if keyword.lower() == "exit":
            return 0

        if not keyword:
            continue

        await _classify_keyword(keyword, db_manager, classifier)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Console-first ad classification")
    parser.add_argument(
        "keyword",
        nargs="?",
        default=None,
        help="Brand keyword to search in the DB and classify",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = _parse_args(argv)
    keyword = args.keyword

    return asyncio.run(_interactive_loop(keyword))


if __name__ == "__main__":
    raise SystemExit(main())
