import numpy as np
from astropy.timeseries import LombScargle

from gaia_variable_stars.config import MIN_POINTS


MIN_PERIOD = 0.1
MAX_PERIOD = 1000.0
SAMPLES_PER_PEAK = 5
N_PEAKS = 5


def period_search(time, mag):
    baseline = np.ptp(time)
    maximum = min(MAX_PERIOD, baseline / 2)
    searchable = len(time) >= MIN_POINTS and np.unique(time).size >= MIN_POINTS and np.std(mag) > 0

    if not searchable or maximum <= MIN_PERIOD:
        return None

    periodogram = LombScargle(time - time.min(), mag, normalization="standard")
    frequency, powers = periodogram.autopower(
        minimum_frequency=1 / maximum,
        maximum_frequency=1 / MIN_PERIOD,
        samples_per_peak=SAMPLES_PER_PEAK,
    )

    valid = (frequency >= 1 / maximum) & (frequency <= 1 / MIN_PERIOD)
    frequency, powers = frequency[valid], powers[valid]
    finite = np.isfinite(powers)

    if not finite.any():
        raise ValueError("No finite Lomb–Scargle powers.")

    available = np.where(finite, powers, -np.inf)
    peaks = [None] * N_PEAKS

    for k in range(N_PEAKS):
        peak = int(np.argmax(available))
        if not np.isfinite(available[peak]):
            break
        peaks[k] = (float(1 / frequency[peak]), float(powers[peak]))
        available[np.abs(frequency - frequency[peak]) <= 1 / baseline] = -np.inf

    peaks = [peak for peak in peaks if peak is not None]
    period, power = peaks[0]

    result = {
        "periodogram": periodogram, "peaks": peaks, "period": period, "power": power,
        "search_max_period": maximum,
        "false_alarm_probability": float(periodogram.false_alarm_probability(
            power, method="baluev", samples_per_peak=SAMPLES_PER_PEAK,
            minimum_frequency=1 / maximum, maximum_frequency=1 / MIN_PERIOD,
        )),
    }

    periods = 1 / frequency

    bands = {
        "short": periods < 1,
        "medium": (periods >= 1) & (periods <= 30),
        "long": periods > 30,
    }

    for name, mask in bands.items():
        indices = np.flatnonzero(mask & finite)
        chosen = indices[np.argmax(powers[indices])] if indices.size else None
        result[f"best_{name}_period"] = float(periods[chosen]) if chosen is not None else np.nan
        result[f"best_{name}_power"] = float(powers[chosen]) if chosen is not None else np.nan

    median = np.median(powers[finite])
    result["peak_ratio_2_1"] = peaks[1][1] / power if len(peaks) > 1 and power > 0 else np.nan
    result["peak_to_median"] = power / median if median > 0 else np.nan
    result["near_search_boundary"] = bool(
        1 / period - 1 / maximum <= 1 / baseline
        or 1 / MIN_PERIOD - 1 / period <= 1 / baseline
    )

    return result


def fourier_fit(time, mag, period, harmonics=(1, 2, 3, 4)):
    if not np.isfinite(period) or period <= 0:
        return None

    phase = 2 * np.pi * ((time - time.min()) / period % 1)
    columns = [np.ones(len(time))] + [
        term for k in harmonics for term in (np.cos(k * phase), np.sin(k * phase))
    ]
    design = np.column_stack(columns)

    if len(time) <= design.shape[1] or np.linalg.matrix_rank(design) != design.shape[1]:
        return None

    coeff = np.linalg.lstsq(design, mag, rcond=None)[0]
    residual = np.sum((mag - design @ coeff) ** 2)
    bic = len(time) * np.log(max(residual / len(time), np.finfo(float).tiny)) + len(coeff) * np.log(len(time))

    amps = {k: np.hypot(*coeff[1 + 2 * i:3 + 2 * i]) for i, k in enumerate(harmonics)}
    phases = {k: np.arctan2(-coeff[2 + 2 * i], coeff[1 + 2 * i]) for i, k in enumerate(harmonics)}
    result = {"coeff": coeff, "bic": bic, "amps": amps, "phases": phases}

    if {1, 2, 3}.issubset(amps) and amps[1] > 0:
        result["R21"] = amps[2] / amps[1]
        result["R31"] = amps[3] / amps[1]
        result["phi21"] = (phases[2] - 2 * phases[1] + np.pi) % (2 * np.pi) - np.pi
        result["phi31"] = (phases[3] - 3 * phases[1] + np.pi) % (2 * np.pi) - np.pi

    return result


def timescale_features(time, mag, errors):
    gaps = np.diff(time)
    delta = np.diff(mag)
    visit = (gaps > 0) & (gaps <= 1)
    pair_error = errors[:-1] ** 2 + errors[1:] ** 2
    adjusted = np.maximum(delta ** 2 - pair_error, 0)
    slopes = np.sqrt(adjusted[visit & np.isfinite(adjusted)]) / gaps[visit & np.isfinite(adjusted)]

    left, right = np.triu_indices(len(time), k=1)
    lags = time[right] - time[left]
    squared = (mag[right] - mag[left]) ** 2
    noise = errors[left] ** 2 + errors[right] ** 2
    usable = np.isfinite(noise)

    short = usable & (lags >= 0.05) & (lags <= 0.3)
    long = usable & (lags > 6)

    sf_short = float(np.mean((squared - noise)[short])) if short.any() else np.nan
    sf_long = float(np.mean((squared - noise)[long])) if long.any() else np.nan

    return {
        "sf_short": sf_short,
        "sf_long": sf_long,
        "sf_ratio": sf_short / sf_long if np.isfinite(sf_short) and np.isfinite(sf_long) and sf_long > 0 else np.nan,
        "max_visit_slope": float(np.max(slopes)) if len(slopes) else np.nan,
    }


def stetson_j(time, mag, errors):
    valid = np.isfinite(errors) & (errors > 0)

    if valid.sum() < 2:
        return np.nan

    normalized = np.full(len(mag), np.nan)
    n = valid.sum()
    mean = np.average(mag[valid], weights=1 / errors[valid] ** 2)
    normalized[valid] = np.sqrt(n / (n - 1)) * (mag[valid] - mean) / errors[valid]

    pairs = (np.diff(time) > 0) & (np.diff(time) <= 0.3)
    product = normalized[:-1] * normalized[1:]
    selected = product[pairs & np.isfinite(product)]

    return float(np.mean(np.sign(selected) * np.sqrt(np.abs(selected)))) if len(selected) else np.nan


def measure(source_id, group):
    time = group["time"].to_numpy(dtype=float)
    mag = group["magnitude"].to_numpy(dtype=float)
    errors = group["magnitude_error"].to_numpy(dtype=float)

    gaps = np.diff(time)
    visit_gaps = gaps[gaps > 1]
    n_visits = int(1 + len(visit_gaps))
    amplitude = group["magnitude"].quantile(0.95) - group["magnitude"].quantile(0.05)
    median_error = group["magnitude_error"].median()

    record = {
        "source_id": source_id,
        "magnitude_mean": mag.mean(),
        "magnitude_std": mag.std(ddof=1) if len(mag) > 1 else np.nan,
        "magnitude_skew": group["magnitude"].skew(),
        "magnitude_kurt": group["magnitude"].kurt(),
        "magnitude_mad": np.median(np.abs(mag - np.median(mag))),
        "amplitude": amplitude,
        "median_error": median_error,
        "amplitude_ratio": amplitude / median_error if np.isfinite(median_error) and median_error > 0 else np.nan,
        "n_epochs": len(time),
        "time_span": np.ptp(time),
        "median_gap": np.median(gaps) if len(gaps) else np.nan,
        "short_gap_fraction": np.mean(gaps < 1) if len(gaps) else np.nan,
        "n_visits": n_visits,
        "epochs_per_visit": len(time) / n_visits,
        "median_visit_gap": np.median(visit_gaps) if len(visit_gaps) else np.nan,
        "visit_gap_cv": np.std(visit_gaps, ddof=1) / np.mean(visit_gaps) if len(visit_gaps) > 1 else np.nan,
        "searchable": False,
        "period_estimate": np.nan,
        "period_fap": np.nan,
        "log_period": np.nan,
        "power": np.nan,
        "half_power": np.nan,
        "double_power": np.nan,
        "half_power_ratio": np.nan,
        "double_power_ratio": np.nan,
        "fourier_amp": np.nan,
        "fourier_ratio": np.nan,
        "search_max_period": np.nan,
        "near_search_boundary": np.nan,
        "peak_ratio_2_1": np.nan,
        "peak_to_median": np.nan,
        "beyond_1sigma_fraction": np.mean(np.abs(mag - np.median(mag)) > mag.std()) if mag.std() > 0 else np.nan,
        "von_neumann_eta": np.mean(np.diff(mag) ** 2) / np.var(mag, ddof=1) if len(mag) > 1 and np.var(mag, ddof=1) > 0 else np.nan,
        "stetson_j": stetson_j(time, mag, errors),
    }

    record.update(timescale_features(time, mag, errors))

    for k in range(1, N_PEAKS + 1):
        record[f"peak_period_{k}"] = np.nan
        record[f"peak_power_{k}"] = np.nan

    for band in ["short", "medium", "long"]:
        record[f"best_{band}_period"] = np.nan
        record[f"best_{band}_power"] = np.nan

    for field in ["R21", "phi21", "R31", "phi31"]:
        record[f"{field}_naive"] = np.nan

    search = period_search(time, mag)

    if search is None:
        return record, (time, mag, errors)

    period = search["period"]
    power = search["power"]
    frequency = 1 / period

    record["searchable"] = True
    record["period_estimate"] = period
    record["period_fap"] = search["false_alarm_probability"]
    record["log_period"] = np.log10(period)
    record["power"] = power
    record["half_power"] = float(search["periodogram"].power(2 * frequency))
    record["double_power"] = float(search["periodogram"].power(frequency / 2))
    record["half_power_ratio"] = record["half_power"] / power if power > 0 else np.nan
    record["double_power_ratio"] = record["double_power"] / power if power > 0 else np.nan
    record["search_max_period"] = search["search_max_period"]
    record["near_search_boundary"] = search["near_search_boundary"]

    for name in ["peak_ratio_2_1", "peak_to_median"]:
        record[name] = search[name]

    for band in ["short", "medium", "long"]:
        record[f"best_{band}_period"] = search[f"best_{band}_period"]
        record[f"best_{band}_power"] = search[f"best_{band}_power"]

    for k, (peak_period, peak_power) in enumerate(search["peaks"], 1):
        record[f"peak_period_{k}"] = peak_period
        record[f"peak_power_{k}"] = peak_power

    naive = fourier_fit(time, mag, period)

    if naive is not None:
        record["fourier_amp"] = naive["amps"].get(1, np.nan)
        record["fourier_ratio"] = naive.get("R21", np.nan)

        for field in ["R21", "phi21", "R31", "phi31"]:
            record[f"{field}_naive"] = naive.get(field, np.nan)

    return record, (time, mag, errors)


def extract(observations):
    results = [measure(source_id, group) for source_id, group in observations.groupby("source_id", sort=False)]
    records = [record for record, curve in results]
    light_curves = {record["source_id"]: curve for record, curve in results}

    return records, light_curves