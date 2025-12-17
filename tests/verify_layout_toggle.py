import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.geometry import GeometryProcessor

def test_layout_toggle():
    print("=== Testing Refactored Camera Layouts ===\n")

    # 1. Ring Mode
    print("1. Testing 'ring' mode with n=6")
    views_ring = GeometryProcessor.generate_views(6, layout_mode='ring')
    names_ring = [v[0] for v in views_ring]
    print(f"   Views: {names_ring}")
    
    # Ring should have View_0 to View_5, no Up/Down
    if "View_5" in names_ring and "Up" not in names_ring:
        print("   PASS: 'ring' mode produced Horizon Ring layout.")
    else:
        print("   FAIL: 'ring' mode incorrect.")

    # 2. Cube Mode
    print("\n2. Testing 'cube' mode (should force 6 views regardless of n)")
    views_cube = GeometryProcessor.generate_views(4, layout_mode='cube') # Pass n=4 to verify it's ignored
    names_cube = [v[0] for v in views_cube]
    print(f"   Views (input n=4): {names_cube}")
    
    expected_cube = ["Front", "Right", "Back", "Left", "Up", "Down"]
    if names_cube == expected_cube:
         print("   PASS: 'cube' mode produced exactly 6 standard views.")
    else:
         print(f"   FAIL: 'cube' mode incorrect. Expected {expected_cube}, got {names_cube}")

    # 3. Fibonacci Mode
    print("\n3. Testing 'fibonacci' mode with n=10")
    views_fib = GeometryProcessor.generate_views(10, layout_mode='fibonacci')
    names_fib = [v[0] for v in views_fib]
    print(f"   Count: {len(views_fib)}")
    print(f"   First view: {views_fib[0]}")
    
    if len(views_fib) == 10 and "View_9" in names_fib:
        print("   PASS: 'fibonacci' mode produced correct number of views.")
    else:
        print("   FAIL: 'fibonacci' mode incorrect.")

if __name__ == "__main__":
    test_layout_toggle()
