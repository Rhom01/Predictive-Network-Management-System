import os
import signal
import sys
import time


from supabase import create_client, Client

from config import (
    DEFAULT_INTERFACE_SPEED_MBPS,
    DEFAULT_PROBE_HOST,
    SAMPLE_INTERVAL,
)

from network_utils import (
    NetworkSampler,
    choose_interface,
    get_default_gateway,
)


running = True


def stop_handler(signum, frame):
    """Gracefully stop the collector."""
    global running
    running = False

    print("\nStopping network collector...", flush=True)


def get_supabase_client() -> Client:
    """Create the Supabase client using environment variables."""

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    if not supabase_url:
        raise RuntimeError(
            "SUPABASE_URL environment variable is missing."
        )

    if not supabase_key:
        raise RuntimeError(
            "SUPABASE_SERVICE_KEY environment variable is missing."
        )

    return create_client(
        supabase_url,
        supabase_key
    )


def clean_value(value):
    """
    Convert values into database-safe values.

    NaN values must become None because PostgreSQL
    does not accept Python NaN in the same way.
    """

    if value is None:
        return None

    try:
        if value != value:
            return None
    except Exception:
        pass

    return value


def prepare_record(row: dict) -> dict:
    """Prepare a sampler row for Supabase."""

    fields = [
        "timestamp",
        "interface",
        "probe_host",
        "bandwidth_mbps",
        "rx_mbps",
        "tx_mbps",
        "throughput_mbps",
        "traffic_volume_mb",
        "packet_rate",
        "latency_ms",
        "packet_loss_pct",
        "jitter_ms",
        "bandwidth_utilization_pct",
        "cpu_utilization_pct",
        "memory_utilization_pct",
        "interface_errors",
        "interface_drops",
        "network_available",
    ]

    record = {}

    for field in fields:
        record[field] = clean_value(row.get(field))

    return record


def send_to_supabase(
    supabase: Client,
    row: dict
):
    """Insert one network measurement into Supabase."""

    record = prepare_record(row)

    response = (
        supabase
        .table("network_metrics")
        .insert(record)
        .execute()
    )

    return response


def display_number(value, decimals=2):
    """Safely format numbers for console output."""

    try:
        if value is None:
            return "N/A"

        value = float(value)

        if value != value:
            return "N/A"

        return f"{value:.{decimals}f}"

    except (TypeError, ValueError):
        return "N/A"


def main():

    global running

    print()
    print("=" * 70)
    print("CLOUD NETWORK COLLECTOR")
    print("=" * 70)

 

    try:

        supabase = get_supabase_client()

        print("Supabase connection : OK")

    except Exception as exc:

        print(
            f"Supabase connection failed: {exc}",
            file=sys.stderr,
            flush=True
        )

        raise SystemExit(1)

    
    interface = choose_interface()

    if not interface:

        raise SystemExit(
            "No usable network interface was found."
        )

 

    probe = (
        get_default_gateway()
        or DEFAULT_PROBE_HOST
    )

    print(
        f"Monitoring interface : {interface}"
    )

    print(
        f"Probe target         : {probe}"
    )

    print(
        f"Sampling interval    : {SAMPLE_INTERVAL} seconds"
    )

    print("=" * 70)

    print(
        "Cloud collector is running..."
    )

    print(
        "Waiting for network measurements..."
    )

    print()

   

    sampler = NetworkSampler(
        interface=interface,
        probe_host=probe,
        fallback_speed_mbps=DEFAULT_INTERFACE_SPEED_MBPS
    )


    signal.signal(
        signal.SIGINT,
        stop_handler
    )

    if hasattr(signal, "SIGTERM"):

        signal.signal(
            signal.SIGTERM,
            stop_handler
        )

 

    sample_number = 0

    while running:

        sample_number += 1

        try:

            row = sampler.sample()

            send_to_supabase(
                supabase,
                row
            )

            timestamp = row.get(
                "timestamp",
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            throughput = row.get(
                "throughput_mbps"
            )

            latency = row.get(
                "latency_ms"
            )

            packet_loss = row.get(
                "packet_loss_pct"
            )

            utilization = row.get(
                "bandwidth_utilization_pct"
            )

            jitter = row.get(
                "jitter_ms"
            )

            print(
                f"[{sample_number}] "
                f"{timestamp} | "
                f"throughput="
                f"{display_number(throughput)} Mbps | "
                f"latency="
                f"{display_number(latency)} ms | "
                f"loss="
                f"{display_number(packet_loss, 1)}% | "
                f"jitter="
                f"{display_number(jitter)} ms | "
                f"util="
                f"{display_number(utilization, 1)}%",
                flush=True
            )

        except Exception as exc:

            print(
                f"[{sample_number}] "
                f"Collector error: {exc}",
                file=sys.stderr,
                flush=True
            )

        time.sleep(SAMPLE_INTERVAL)

    print()
    print("=" * 70)
    print(
        f"Collector stopped after "
        f"{sample_number} samples."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()