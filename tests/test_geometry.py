import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from core.geometry import GeometryProcessor

def test_rotation_matrix():
    print("Testing Rotation Matrix...")
    # Test identity (0,0,0)
    R = GeometryProcessor.get_rotation_matrix(0, 0, 0)
    assert np.allclose(R, np.eye(3)), "Rotation 0,0,0 should be identity"
    
    # Test Yaw 90 (Rotation around Y)
    # Ry(90) should map Z (0,0,1) to X (1,0,0) if we consider vector rotation v' = R v
    # The code uses v' = v @ R.T which is equivalent to v' = (R @ v.T).T
    
    # Let's check the matrix explicitly for Yaw 90
    R_90_yaw = GeometryProcessor.get_rotation_matrix(90, 0, 0)
    
    # Expected Ry(90)
    # [ cos(90)  0  sin(90)]   [ 0  0  1]
    # [    0     1     0   ] = [ 0  1  0]
    # [-sin(90)  0  cos(90)]   [-1  0  0]
    
    expected = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    
    assert np.allclose(R_90_yaw, expected, atol=1e-7), "Rotation Yaw 90 incorrect"
    print("Rotation Matrix Passed.")

def test_map_generation_shapes():
    print("Testing Map Generation Shapes...")
    src_h, src_w = 2000, 4000
    dest_h, dest_w = 512, 512
    fov = 90
    
    # Front view
    map_x, map_y = GeometryProcessor.create_rectilinear_map(
        src_h, src_w, dest_h, dest_w, fov, 0, 0, 0
    )
    
    assert map_x.shape == (dest_h, dest_w)
    assert map_y.shape == (dest_h, dest_w)
    assert not np.isnan(map_x).any()
    assert not np.isnan(map_y).any()
    print("Map Generation Shapes Passed.")

def test_map_generation_values():
    print("Testing Map Generation Values...")
    src_h, src_w = 1000, 2000 # 2:1 aspect ratio
    dest_h, dest_w = 100, 100
    fov = 90
    
    # 1. Test Center View (0,0,0)
    # The center pixel of the destination should map to the center of the source
    map_x, map_y = GeometryProcessor.create_rectilinear_map(
        src_h, src_w, dest_h, dest_w, fov, 0, 0, 0
    )
    
    center_idx = (dest_h // 2, dest_w // 2)
    center_map_x = map_x[center_idx]
    center_map_y = map_y[center_idx]
    
    expected_x = src_w / 2
    expected_y = src_h / 2
    
    # Allow small floating point error
    assert np.isclose(center_map_x, expected_x, atol=1.0), \
        f"Center View: Expected x={expected_x}, got {center_map_x}"
    assert np.isclose(center_map_y, expected_y, atol=1.0), \
        f"Center View: Expected y={expected_y}, got {center_map_y}"

    # 2. Test Right View (Yaw=90)
    # Looking 90 deg right. Center of dest should map to 75% of source width (theta=pi/2 -> 0.75)
    map_x, map_y = GeometryProcessor.create_rectilinear_map(
        src_h, src_w, dest_h, dest_w, fov, 90, 0, 0
    )
    
    center_map_x = map_x[center_idx]
    center_map_y = map_y[center_idx]
    
    # Theta ranges from -pi to pi.
    # 0 -> 0.5 * W
    # pi/2 -> (0.25 + 0.5) * W = 0.75 * W
    expected_x = 0.75 * src_w
    expected_y = src_h / 2 # Pitch is 0, so still middle height
    
    assert np.isclose(center_map_x, expected_x, atol=1.0), \
        f"Right View: Expected x={expected_x}, got {center_map_x}"
    assert np.isclose(center_map_y, expected_y, atol=1.0), \
        f"Right View: Expected y={expected_y}, got {center_map_y}"

    # 3. Test Back View (Yaw=180)
    # Looking back. Center of dest should map to 0% or 100% (wrap around) of source width.
    # Actually code logic: theta=pi -> (0.5 + 0.5)=1.0, theta=-pi -> (-0.5+0.5)=0.0
    # atan2(x, z) for Yaw=180:
    # R(180) = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    # v=[0,0,1] -> v' = [-1, 0, -1] ? No.
    # v=[0,0,1] -> R*v = [-1, 0, 0] * 0 + [0,1,0]*0 + [0,0,-1]*1 = [0, 0, -1]
    # x=0, z=-1. atan2(0, -1) = pi (or -pi).
    # map to 1.0 * src_w or 0.0 * src_w.
    
    map_x, map_y = GeometryProcessor.create_rectilinear_map(
        src_h, src_w, dest_h, dest_w, fov, 180, 0, 0
    )
    center_map_x = map_x[center_idx]
    
    # It might be 0 or src_w (2000).
    # Let's check if it's close to either boundary.
    assert np.isclose(center_map_x, 0, atol=1.0) or np.isclose(center_map_x, src_w, atol=1.0), \
        f"Back View: Expected x~0 or {src_w}, got {center_map_x}"

    print("Map Generation Values Passed.")

def test_generate_views_always_ring():
    print("Testing Generate Views (Always Ring)...")
    
    # Test small n
    views_3 = GeometryProcessor.generate_views(3, pitch_offset=0)
    assert len(views_3) == 3
    assert views_3[0][1] == 0.0
    assert views_3[1][1] == 120.0
    assert views_3[2][1] == 240.0
    
    # Test n=6 (previously Cube)
    views_6 = GeometryProcessor.generate_views(6, pitch_offset=10)
    assert len(views_6) == 6
    for i, (name, yaw, pitch, roll) in enumerate(views_6):
        assert pitch == 10
        assert np.isclose(yaw, i * 60.0)
        
    # Test n=10 (previously Fibonacci)
    views_10 = GeometryProcessor.generate_views(10, pitch_offset=-15)
    assert len(views_10) == 10
    for i, (name, yaw, pitch, roll) in enumerate(views_10):
        assert pitch == -15
        assert np.isclose(yaw, i * 36.0)
        
    print("Generate Views (Always Ring) Passed.")

if __name__ == "__main__":
    try:
        test_rotation_matrix()
        test_map_generation_shapes()
        test_map_generation_values()
        test_generate_views_always_ring()
        print("All Geometry Tests Passed!")
    except AssertionError as e:
        print(f"Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)