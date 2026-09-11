import platform
import re
import subprocess
import time
from collections import deque
from typing import Optional

import numpy as np
import psutil




def list_interfaces():
    """
    Return all network interfaces detected by the operating system.
    """
    return list(psutil.net_if_addrs().keys())


def get_interface_speed_mbps(
    interface: str,
    fallback: float = 100.0,
) -> float:
    """
    Return the reported link speed of a network interface in Mbps.

    If the operating system does not provide a valid speed,
    the fallback value is returned.
    """
    try:
        stats = psutil.net_if_stats().get(interface)

        if stats is not None:
            speed = float(stats.speed)

            if np.isfinite(speed) and speed > 0:
                return speed

    except Exception:
        pass

    return float(fallback)


def _is_virtual_interface(name: str) -> bool:
    """
    Identify interfaces that are normally virtual/non-Internet adapters.
    """

    lname = name.lower()

    virtual_keywords = [
        "loopback",
        "pseudo",
        "vethernet",
        "default switch",
        "hyper-v",
        "vmware",
        "virtualbox",
        "virtual",
        "docker",
        "wsl",
        "tailscale",
        "zerotier",
        "vpn",
        "tap",
        "tun",
        "teredo",
        "isatap",
        "local area connection*",
    ]

    return any(keyword in lname for keyword in virtual_keywords)


def _interface_score(
    name: str,
    info,
    counters,
) -> float:
    """
    Give an active network interface a score.

    Physical Internet interfaces are deliberately preferred over
    virtual adapters.
    """

    lname = name.lower()

    score = 0.0

    if "wifi" in lname:
        score += 200

    elif "wireless" in lname:
        score += 180

 
    elif lname == "ethernet" or "ethernet" in lname:
        score += 160

   
    if _is_virtual_interface(name):
        score -= 500


    if counters is not None:

        total_bytes = (
            float(counters.bytes_sent)
            + float(counters.bytes_recv)
        )

        total_packets = (
            float(counters.packets_sent)
            + float(counters.packets_recv)
        )

        if total_bytes > 0:
            score += 50

        if total_packets > 0:
            score += 25

        score += min(total_bytes / 1_000_000_000.0, 25.0)

    
    try:
        if info.speed and info.speed > 0:
            score += 10
    except Exception:
        pass

    return score


def choose_interface(preferred: Optional[str] = None) -> str:
    """
    Automatically select the most appropriate network interface.

    Priority:

        1. Explicitly requested interface
        2. Active WiFi/Ethernet carrying traffic
        3. Other physical-looking active interface
        4. Last available interface

    This prevents interfaces such as:

        vEthernet (Default Switch)
        Loopback Pseudo-Interface 1
        VMware
        VirtualBox

    from being selected instead of the real Internet interface.
    """

    interfaces = psutil.net_if_addrs()
    stats = psutil.net_if_stats()
    counters = psutil.net_io_counters(pernic=True)


    if preferred:
        if preferred not in interfaces:
            raise ValueError(
                f"Requested network interface '{preferred}' "
                f"was not found.\n\n"
                f"Available interfaces:\n"
                f"{', '.join(interfaces.keys())}"
            )

        interface_stats = stats.get(preferred)

        if interface_stats and interface_stats.isup:
            return preferred

        raise RuntimeError(
            f"Requested network interface '{preferred}' "
            f"is currently down."
        )

   

    candidates = []

    for name, info in stats.items():

        if not info.isup:
            continue

        interface_counters = counters.get(name)

        score = _interface_score(
            name=name,
            info=info,
            counters=interface_counters,
        )

        candidates.append(
            (
                score,
                name,
            )
        )



    if candidates:

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return candidates[0][1]



    if interfaces:
        return next(iter(interfaces))

    return ""




def get_default_gateway_windows() -> Optional[str]:
    """
    Get the default IPv4 gateway on Windows.
    """

    try:
        output = subprocess.check_output(
            ["ipconfig"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

        for line in output.splitlines():

            if "Default Gateway" not in line:
                continue

            value = line.split(":", 1)[-1].strip()

            if not value:
                continue

          
            if ":" in value:
                continue

            if re.match(
                r"^\d{1,3}(?:\.\d{1,3}){3}$",
                value,
            ):
                return value

    except Exception:
        pass

    return None


def get_default_gateway_linux() -> Optional[str]:
    """
    Get the default IPv4 gateway on Linux.
    """

    try:
        output = subprocess.check_output(
            ["ip", "route"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

        match = re.search(
            r"default\s+via\s+([0-9.]+)",
            output,
        )

        if match:
            return match.group(1)

    except Exception:
        pass

    return None


def get_default_gateway() -> Optional[str]:
    """
    Detect the default gateway according to the operating system.
    """

    system = platform.system().lower()

    if system.startswith("win"):
        return get_default_gateway_windows()

    return get_default_gateway_linux()




def ping_once(
    host: str,
    timeout_seconds: float = 2.0,
):
    """
    Perform one ICMP ping.

    Returns:

        (latency_ms, reachable)

    Example:

        (24.6, True)

    If unreachable:

        (None, False)
    """

    system = platform.system().lower()

    if system.startswith("win"):

        command = [
            "ping",
            "-n",
            "1",
            "-w",
            str(int(timeout_seconds * 1000)),
            host,
        ]

    else:

        command = [
            "ping",
            "-c",
            "1",
            "-W",
            str(max(1, int(timeout_seconds))),
            host,
        ]

    started = time.perf_counter()

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout_seconds + 1.0,
        )

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        if result.returncode != 0:
            return None, False

        text = (
            result.stdout
            + result.stderr
        )

        # Windows / Linux reported RTT.
        match = re.search(
            r"time[=<]\s*"
            r"([0-9]+(?:\.[0-9]+)?)"
            r"\s*ms",
            text,
            re.IGNORECASE,
        )

        if match:
            return float(match.group(1)), True

        # Fallback to measured elapsed time.
        return elapsed_ms, True

    except Exception:
        return None, False




class NetworkSampler:

    def __init__(
        self,
        interface: str,
        probe_host: str,
        fallback_speed_mbps: float,
    ):
        """
        Create a network sampler.

        Parameters
        ----------
        interface:
            Network interface to monitor.

        probe_host:
            IP/hostname used for latency testing.

        fallback_speed_mbps:
            Used if Windows/Linux does not report
            the interface speed.
        """

        self.interface = interface
        self.probe_host = probe_host

  

        self.speed_mbps = get_interface_speed_mbps(
            interface,
            fallback=fallback_speed_mbps,
        )

     

        self.previous = None
        self.last_time = None

     

        self.rtt_history = deque(
            maxlen=12
        )

  

    def sample(self):
        """
        Collect one network performance sample.

        Returns a dictionary compatible with collector.py,
        train.py and app.py.
        """

        now = time.time()

   

        counters = (
            psutil
            .net_io_counters(pernic=True)
            .get(self.interface)
        )

        if counters is None:

            raise RuntimeError(
                f"Network interface "
                f"'{self.interface}' was not found."
            )

        

        cpu = psutil.cpu_percent(
            interval=None
        )

        memory = psutil.virtual_memory().percent

    

        rtt, reachable = ping_once(
            self.probe_host
        )

        if (
            reachable
            and rtt is not None
            and np.isfinite(rtt)
        ):
            self.rtt_history.append(
                float(rtt)
            )

      

        if len(self.rtt_history) >= 2:

            jitter = float(
                np.std(
                    self.rtt_history
                )
            )

        else:

            jitter = np.nan

   

        rx_bytes = float(
            counters.bytes_recv
        )

        tx_bytes = float(
            counters.bytes_sent
        )

        rx_packets = float(
            counters.packets_recv
        )

        tx_packets = float(
            counters.packets_sent
        )

        total_errors = float(
            counters.errin
            + counters.errout
        )

        total_drops = float(
            counters.dropin
            + counters.dropout
        )

   

        rx_bps = np.nan
        tx_bps = np.nan
        total_bps = np.nan
        packet_rate = np.nan

      

        if (
            self.previous is not None
            and self.last_time is not None
        ):

            dt = max(
                now - self.last_time,
                0.001,
            )

            previous_rx = self.previous[0]
            previous_tx = self.previous[1]
            previous_packets = self.previous[2]


            rx_delta = (
                rx_bytes - previous_rx
            )

            tx_delta = (
                tx_bytes - previous_tx
            )

            packet_delta = (
                (rx_packets + tx_packets)
                - previous_packets
            )

          

            if rx_delta < 0:
                rx_delta = 0.0

            if tx_delta < 0:
                tx_delta = 0.0

            if packet_delta < 0:
                packet_delta = 0.0

          

            rx_bps = (
                rx_delta
                * 8.0
                / dt
            )

            tx_bps = (
                tx_delta
                * 8.0
                / dt
            )

            total_bps = (
                (rx_delta + tx_delta)
                * 8.0
                / dt
            )

            packet_rate = (
                packet_delta
                / dt
            )


        utilization = np.nan

        if (
            self.speed_mbps > 0
            and np.isfinite(total_bps)
        ):

            utilization = (
                total_bps
                / (
                    self.speed_mbps
                    * 1_000_000.0
                )
            ) * 100.0

          
            utilization = max(
                0.0,
                min(
                    float(utilization),
                    100.0,
                ),
            )


        packet_loss = (
            0.0
            if reachable
            else 100.0
        )

       

        rx_mbps = (
            rx_bps / 1_000_000.0
            if np.isfinite(rx_bps)
            else np.nan
        )

        tx_mbps = (
            tx_bps / 1_000_000.0
            if np.isfinite(tx_bps)
            else np.nan
        )

        throughput_mbps = (
            total_bps / 1_000_000.0
            if np.isfinite(total_bps)
            else np.nan
        )


        if np.isfinite(total_bps):

            traffic_volume_mb = (
                total_bps
                / 8.0
                / 1_000_000.0
            )

        else:

            traffic_volume_mb = np.nan

     

        row = {

            "timestamp": now,

            "interface": self.interface,

            "probe_host": self.probe_host,

            "bandwidth_mbps": self.speed_mbps,

            "rx_mbps": rx_mbps,

            "tx_mbps": tx_mbps,

            "throughput_mbps": throughput_mbps,

            "traffic_volume_mb": traffic_volume_mb,

            "packet_rate": packet_rate,

            "latency_ms": (
                float(rtt)
                if rtt is not None
                else np.nan
            ),

            "packet_loss_pct": packet_loss,

            "jitter_ms": jitter,

            "bandwidth_utilization_pct": utilization,

            "cpu_utilization_pct": cpu,

            "memory_utilization_pct": memory,

            "interface_errors": total_errors,

            "interface_drops": total_drops,

            "network_available": int(
                reachable
            ),
        }

    

        self.previous = (
            rx_bytes,
            tx_bytes,
            rx_packets + tx_packets,
        )

        self.last_time = now

        return row