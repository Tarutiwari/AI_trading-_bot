import requests
import pandas as pd
from datetime import datetime, timedelta
import time

START_DATE = "2026-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

OUTPUT_FILE = "data/forex_factory_calendar.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    )
}


def fetch_week(start_date, end_date):
    """
    Fetch economic calendar data for a date range.
    Forex Factory commonly exposes weekly calendar data
    through this endpoint.
    """

    start_str = start_date.strftime("%m-%d-%Y")
    end_str = end_date.strftime("%m-%d-%Y")

    url = (
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        f"?start={start_str}&end={end_str}"
    )

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )

        response.raise_for_status()
        return response.json()

    except Exception as e:
        print(f"❌ Error fetching {start_date} -> {end_date}: {e}")
        return []


def main():
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")

    all_events = []
    current = start

    print(f"🚀 Fetching Forex Factory calendar...")
    print(f"📅 From {START_DATE} to {END_DATE}\n")

    while current <= end:

        week_end = min(
            current + timedelta(days=6),
            end
        )

        print(
            f"Fetching: "
            f"{current.strftime('%Y-%m-%d')} "
            f"→ "
            f"{week_end.strftime('%Y-%m-%d')}"
        )

        events = fetch_week(current, week_end)

        if events:
            all_events.extend(events)
            print(f"   ✅ {len(events)} events")

        else:
            print("   ⚠️ No events returned")

        current = week_end + timedelta(days=1)

        # Don't spam requests
        time.sleep(1)

    if not all_events:
        print("\n❌ No data collected.")
        return

    df = pd.DataFrame(all_events)

    print("\n📊 Total events:", len(df))
    print("Columns:", df.columns.tolist())

    # Convert date column if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
            utc=True
        )

    # Keep only useful currencies for XAUUSD
    important_currencies = ["USD", "EUR", "GBP", "JPY", "CNY"]

    if "country" in df.columns:
        df = df[
            df["country"].isin(important_currencies)
        ]

    # Sort chronologically
    if "date" in df.columns:
        df = df.sort_values("date")

    # Remove duplicates
    df = df.drop_duplicates()

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\n🎉 Saved {len(df)} events")
    print(f"📁 File: {OUTPUT_FILE}")

    print("\nSample:")
    print(df.head())


if __name__ == "__main__":
    main()