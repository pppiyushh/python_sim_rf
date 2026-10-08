"""A first 16-QAM transmitter/channel/receiver simulation.

Install: python -m pip install numpy matplotlib
Run:     python qam16_link.py

This is a complex-baseband, one-sample-per-symbol simulation with ideal
timing and carrier synchronization. It has no coding, pulse shaping,
fading, or OFDM yet. SNR means Es/N0 (symbol energy / noise density).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Gray mapping on either axis: adjacent levels differ by one bit.
# Bits:        00  01  11  10
# Amplitude:   -3  -1  +1  +3
# Divide by sqrt(10) to make the average energy of uniform 16-QAM equal 1.
SCALE = np.sqrt(10.0)
LEVELS = np.array([-3, -1, 1, 3]) / SCALE
LEVEL_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
TX_LOOKUP = np.array([-3, -1, 3, 1]) / SCALE  # Indexed by 00, 01, 10, 11.


def transmitter(bits):
    """Map each row [I_bit0, I_bit1, Q_bit0, Q_bit1] to one symbol."""
    i_index = 2 * bits[:, 0] + bits[:, 1]
    q_index = 2 * bits[:, 2] + bits[:, 3]
    return TX_LOOKUP[i_index] + 1j * TX_LOOKUP[q_index]


def awgn_channel(tx_symbols, snr_db, rng):
    """Add independent Gaussian noise to I and Q, using nominal Es = 1."""
    es_over_n0 = 10.0 ** (snr_db / 10.0)
    # E[|noise|^2] = N0 = 1 / (Es/N0). Each real component gets N0/2.
    sigma = np.sqrt(1.0 / (2.0 * es_over_n0))
    noise = sigma * (
        rng.normal(size=tx_symbols.size) + 1j * rng.normal(size=tx_symbols.size)
    )
    return tx_symbols + noise


def receiver(rx_symbols):
    """Select the nearest amplitude on each axis and recover four bits."""
    i_index = np.argmin(abs(rx_symbols.real[:, None] - LEVELS), axis=1)
    q_index = np.argmin(abs(rx_symbols.imag[:, None] - LEVELS), axis=1)
    return np.column_stack((LEVEL_BITS[i_index], LEVEL_BITS[q_index]))


def plot_constellation(tx_symbols, rx_symbols, symbol_errors, snr_db, ber, output):
    """Show the ideal constellation and the first 2,000 received symbols."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    fig.suptitle("16-QAM: transmitter → noisy channel → receiver", fontsize=17)

    # All 16 ideal points, labelled with the four bits represented by each.
    for i, i_level in enumerate(LEVELS):
        for q, q_level in enumerate(LEVELS):
            label = "".join(str(int(bit)) for bit in (*LEVEL_BITS[i], *LEVEL_BITS[q]))
            axes[0].scatter(i_level, q_level, s=75, color="#176baf")
            axes[0].annotate(label, (i_level, q_level), xytext=(0, 9),
                             textcoords="offset points", ha="center", fontsize=11)
    axes[0].set_title("Transmitter: 4 bits per symbol", fontsize=14)

    count = min(2000, rx_symbols.size)
    points = rx_symbols[:count]
    wrong = symbol_errors[:count]
    axes[1].scatter(points.real[~wrong], points.imag[~wrong], s=9,
                    alpha=0.35, color="#176baf", label="Correct symbol")
    axes[1].scatter(points.real[wrong], points.imag[wrong], s=18,
                    alpha=0.8, color="#c43e3e", label="Incorrect symbol")
    axes[1].scatter(tx_symbols[:count].real, tx_symbols[:count].imag,
                    marker="+", s=80, linewidths=1.5, color="black",
                    label="Ideal points")
    axes[1].set_title(f"Receiver: Es/N0 = {snr_db:g} dB · BER = {ber:.4g}",
                      fontsize=14)
    axes[1].legend(loc="upper right", fontsize=10)

    # The nearest-point receiver's decision boundaries.
    for ax in axes:
        for threshold in [-2 / SCALE, 0, 2 / SCALE]:
            ax.axvline(threshold, color="gray", linestyle="--", alpha=0.45)
            ax.axhline(threshold, color="gray", linestyle="--", alpha=0.45)
        ax.set(xlabel="In-phase (I)", ylabel="Quadrature (Q)")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=11)
    limit = max(1.5, float(np.max(np.abs(points.real))),
                float(np.max(np.abs(points.imag)))) + 0.05
    for ax in axes:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

    fig.savefig(output, dpi=180)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=10000)
    parser.add_argument("--snr-db", type=float, default=14.0,
                        help="Es/N0 in dB (default: 14)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("constellation_16qam.png"))
    parser.add_argument("--no-show", action="store_true",
                        help="Save the plot without opening a window")
    args = parser.parse_args()
    if args.symbols < 1:
        parser.error("--symbols must be positive")
    if not np.isfinite(args.snr_db) or not -100 <= args.snr_db <= 100:
        parser.error("--snr-db must be finite and between -100 and 100")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")

    rng = np.random.default_rng(args.seed)
    tx_bits = rng.integers(0, 2, size=(args.symbols, 4), dtype=np.uint8)
    tx_symbols = transmitter(tx_bits)
    rx_symbols = awgn_channel(tx_symbols, args.snr_db, rng)
    rx_bits = receiver(rx_symbols)

    bit_errors = int(np.count_nonzero(tx_bits != rx_bits))
    symbol_errors = np.any(tx_bits != rx_bits, axis=1)
    ber = bit_errors / tx_bits.size
    ser = float(np.mean(symbol_errors))

    print(f"Symbols: {args.symbols:,}; bits: {tx_bits.size:,}; Es/N0: {args.snr_db:g} dB")
    print(f"Bit errors: {bit_errors:,}; BER: {ber:.6f}")
    print(f"Symbol errors: {np.count_nonzero(symbol_errors):,}; SER: {ser:.6f}")
    print("\nFirst 8 symbols (normalized I + jQ):")
    print(" TX bits     TX symbol          RX symbol       RX bits")
    for sent_bits, sent, received, decoded in zip(
            tx_bits[:8], tx_symbols[:8], rx_symbols[:8], rx_bits[:8]):
        sent_word = "".join(str(int(bit)) for bit in sent_bits)
        rx_word = "".join(str(int(bit)) for bit in decoded)
        print(f" {sent_word}     {sent.real:+.3f}{sent.imag:+.3f}j"
              f"     {received.real:+.3f}{received.imag:+.3f}j     {rx_word}")

    fig = plot_constellation(tx_symbols, rx_symbols, symbol_errors,
                             args.snr_db, ber, args.output)
    print(f"\nPlot saved to: {args.output.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
