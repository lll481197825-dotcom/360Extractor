# 360 Extractor: "Road to Release" Master Plan

This document outlines the strategic roadmap to elevate 360 Extractor from a functional prototype to a commercial-grade, public-ready product. It focuses on Professional Image Quality, polished User Experience (UX), and streamlined Deployment.

---

## Tier 1: Professional Quality (Image Processing)
**Goal:** Ensure the output data is not just "cut" but "enhanced" for photogrammetry, minimizing downstream errors in tools like RealityCapture.

### 1.1 Advanced Image Enhancement Pipeline
*   **Conservative Sharpening (Optional):**
    *   **Context:** Rectilinear projection stretches pixels, causing slight softness.
    *   **Feature:** Apply Unsharp Masking (USM) or Laplacian sharpening kernel after re-projection.
    *   **Constraint:** Must be **optional** (default OFF). Extensive testing required to ensure no artifacts are introduced that could confuse structure-from-motion (SFM) algorithms. Focus on recovering details without altering pixel truth.
    *   **Control:** "Sharpness" slider (0.0 - 2.0).

### 1.2 Performance & Acceleration
*   **GPU Acceleration for OpenCV:**
    *   **Context:** Current `cv2.remap` runs on CPU.
    *   **Improvement:** Utilize OpenCV's `UMat` (Transparent API) to leverage OpenCL on macOS/Windows GPUs automatically without requiring a full CUDA build.
*   **Parallel Extraction:**
    *   **Context:** `VideoProcessor` handles one video at a time.
    *   **Improvement:** Allow concurrent processing of 2-3 videos on high-core-count machines (e.g., M1/M2/M3 Max).

---

## Tier 2: User Experience Polish (UI/UX)
**Goal:** Make the application feel "native," "safe," and "informative" for non-technical users.

### 2.1 "Premium" Interaction Details
*   **Rich Tooltips:**
    *   Add hover tooltips to every setting (e.g., explain "Fibonacci Sphere" vs. "Cube" visually or with text).
*   **Status Bar Telemetry:**
    *   **Current:** Shows "Processing...".
    *   **Target:** Show "Processing [Filename] - Frame 45/900 - ETA: 2m 15s - GPU: Active".
*   **Empty State / Welcome Screen:**
    *   When the queue is empty, the list should not just be blank. It should show a "Getting Started" guide or illustration encouraging the drag-and-drop action.

### 2.2 Workflow Refinements
*   **Drag-and-Drop Sorting:**
    *   Allow users to reorder the job queue by dragging items.
*   **Job History:**
    *   Keep a log of "Completed Jobs" with a "Show in Finder/Explorer" button.
*   **Help & Documentation:**
    *   Add a `Help` menu with:
        *   "User Manual" (PDF/Link).
        *   "Report Issue" link.
        *   "About" dialog with version info.

---

## Tier 3: Deployment & Maintenance
**Goal:** Ensure the source code is easy to set up and maintain for developers and users running from source.

### 3.1 Source Distribution
*   **Repository:** Maintain a clean, well-documented repository.
*   **Dependencies:** Keep `requirements.txt` up-to-date and minimal.
*   **Setup:** Provide clear instructions for setting up the Python environment (virtualenv, conda).

### 3.2 Updates & Maintenance
*   **Version Checking:**
    *   Optional: Check a remote `version.json` to notify the user of updates.
*   **Crash Reporting:**
    *   Integrate Sentry (Python SDK) to automatically report unhandled exceptions (Optional/Opt-in).