import argparse
import asyncio
import json
import sys

from config import get_settings
from database import DatabaseManager
from model import AdClassifier


async def _run_once(keyword: str) -> int:
    settings = get_settings()

    if not keyword or not keyword.strip():
        print("ERROR: keyword is required", file=sys.stderr)
        return 2

    keyword = keyword.strip()

    print("Initializing...")
    db_manager = DatabaseManager(settings.database_path)
    classifier = AdClassifier()
    classifier.load_model()

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

    if keyword is None:
        try:
            keyword = input("Enter keyword: ").strip()
        except EOFError:
            keyword = ""

    return asyncio.run(_run_once(keyword))


if __name__ == "__main__":
    raise SystemExit(main())
