#!/usr/bin/env python3
"""Lock Arducam OV9782 white balance to 5600K via AVFoundation.

The Arducam shows up as AVCaptureDALDevice which has a slightly different
API surface than standard AVCaptureDevice. Key method names use
'deviceWhiteBalanceGains' instead of 'whiteBalanceGains'.
"""

import AVFoundation
import objc
from Foundation import NSObject


def find_arducam():
    """Find the Arducam device among all available capture devices."""
    devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
    for dev in devices:
        name = dev.localizedName()
        print(f"  Found device: {name}")
        if "Arducam" in name or "OV9782" in name:
            print(f"  -> Selected: {name}")
            return dev
    for dev in devices:
        name = dev.localizedName()
        if "FaceTime" not in name:
            return dev
    return None


def lock_white_balance(device, target_temp_k=5600):
    """Lock white balance to target temperature in Kelvin."""
    print(f"\n[WB] Target: {target_temp_k}K")

    # Check mode support
    locked_supported = device.isWhiteBalanceModeSupported_(AVFoundation.AVCaptureWhiteBalanceModeLocked)
    custom_gains_supported = device.isLockingWhiteBalanceWithCustomDeviceGainsSupported()
    print(f"[WB] Locked mode supported: {locked_supported}")
    print(f"[WB] Custom gains supported: {custom_gains_supported}")

    # Get current state
    mode = device.whiteBalanceMode()
    mode_name = {0: "ContinuousAuto", 1: "Locked"}.get(mode, f"Unknown({mode})")
    print(f"[WB] Current mode: {mode_name}")

    # Get current gains
    current_gains = device.deviceWhiteBalanceGains()
    print(f"[WB] Current gains: R={current_gains.redValue:.4f} G={current_gains.greenValue:.4f} B={current_gains.blueValue:.4f}")

    # Get current temp/tint
    current_tt = device.temperatureAndTintValuesForDeviceWhiteBalanceGains_(current_gains)
    print(f"[WB] Current temp: {current_tt.temperature:.0f}K  tint: {current_tt.tint:.1f}")

    # Try the temperature+tint direct method first — this is the most
    # direct path: it computes the gains internally and locks in one call.
    # Temperature 5600K, tint 0 (neutral).
    target_tt = AVFoundation.AVCaptureWhiteBalanceTemperatureAndTintValues(
        temperature=target_temp_k,
        tint=0.0
    )

    # Compute gains for 5600K to show what we're setting
    target_gains = device.deviceWhiteBalanceGainsForTemperatureAndTintValues_(target_tt)
    print(f"[WB] Computed gains for {target_temp_k}K: R={target_gains.redValue:.4f} G={target_gains.greenValue:.4f} B={target_gains.blueValue:.4f}")

    # Clamp to device max
    max_gain = device.maxWhiteBalanceGain()
    clamped = AVFoundation.AVCaptureWhiteBalanceGains(
        redValue=min(max(0.0, target_gains.redValue), max_gain),
        greenValue=min(max(0.0, target_gains.greenValue), max_gain),
        blueValue=min(max(0.0, target_gains.blueValue), max_gain),
    )

    # Lock device for configuration
    error = objc.nil
    ok = device.lockForConfiguration_(error)
    if not ok[0]:
        print(f"[WB] Failed to lock device: {ok[1]}")
        return False

    try:
        # Method 1: Temperature + Tint (most direct)
        print(f"[WB] Calling setWhiteBalanceModeLockedWithTemperatureAndTint...")
        device.setWhiteBalanceModeLockedWithDeviceWhiteBalanceTemperatureAndTintValues_completionHandler_(
            target_tt, None
        )

        # Verify
        new_mode = device.whiteBalanceMode()
        new_gains = device.deviceWhiteBalanceGains()
        new_tt = device.temperatureAndTintValuesForDeviceWhiteBalanceGains_(new_gains)
        new_mode_name = {0: "ContinuousAuto", 1: "Locked"}.get(new_mode, f"Unknown({new_mode})")
        print(f"[WB] After lock:")
        print(f"[WB]   Mode: {new_mode_name}")
        print(f"[WB]   Gains: R={new_gains.redValue:.4f} G={new_gains.greenValue:.4f} B={new_gains.blueValue:.4f}")
        print(f"[WB]   Temp: {new_tt.temperature:.0f}K  tint: {new_tt.tint:.1f}")

        if new_mode == 1:  # Locked
            print(f"\n[SUCCESS] Hardware white balance locked to {target_temp_k}K")
            return True

        # Method 2: Direct gains
        print(f"[WB] Temp+Tint didn't lock. Trying direct gains...")
        device.setWhiteBalanceModeLockedWithDeviceWhiteBalanceGains_completionHandler_(
            clamped, None
        )
        new_mode = device.whiteBalanceMode()
        new_gains = device.deviceWhiteBalanceGains()
        new_tt = device.temperatureAndTintValuesForDeviceWhiteBalanceGains_(new_gains)
        new_mode_name = {0: "ContinuousAuto", 1: "Locked"}.get(new_mode, f"Unknown({new_mode})")
        print(f"[WB] After gains lock:")
        print(f"[WB]   Mode: {new_mode_name}")
        print(f"[WB]   Gains: R={new_gains.redValue:.4f} G={new_gains.greenValue:.4f} B={new_gains.blueValue:.4f}")
        print(f"[WB]   Temp: {new_tt.temperature:.0f}K  tint: {new_tt.tint:.1f}")

        if new_mode == 1:
            print(f"\n[SUCCESS] Hardware white balance locked to {target_temp_k}K (via direct gains)")
            return True

        print(f"\n[WB] Hardware lock not accepted. Mode stays {new_mode_name}.")
        print(f"[WB] The OV9782 sensor may not support hardware WB lock.")
        print(f"[WB] Applying software WB correction in capture pipeline instead.")
        return "software"

    finally:
        device.unlockForConfiguration()


def main():
    print("=== AVFoundation White Balance Lock ===\n")
    device = find_arducam()
    if device is None:
        print("[ERROR] No Arducam found")
        return False

    print(f"\nDevice: {device.localizedName()}")
    print(f"Class: {device.className()}")

    result = lock_white_balance(device, target_temp_k=5600)
    return result


if __name__ == "__main__":
    main()
