import torch

def compute_normal_vector(world_points):
    vector1 = world_points[1] - world_points[0]
    vector2 = world_points[2] - world_points[0]
    normal = torch.cross(vector1, vector2, dim=0)
    normal_normalized = normal / torch.norm(normal)
    return normal_normalized

def transform_points_to_world(p, camera_data):
    batch_size = camera_data.shape[0]
    device = p.device  # Assuming p and camera_data are on the same device

    normals = []

    for i in range(batch_size):
        P = camera_data[i, -16:].reshape(4, 4).to(device)
        
        # Initialize K with gradients disabled as it's a constant matrix
        K = torch.zeros((3, 3), device=device, requires_grad=False)
        K[0, 0] = camera_data[i, 16]
        K[1, 1] = camera_data[i, 17]
        K[0, 2] = camera_data[i, 18]
        K[1, 2] = camera_data[i, 19]
        K[2, 2] = 1.0

        K_inv = torch.inverse(K)

        # Use slicing and operations that maintain gradient tracking
        pixel_points = torch.cat([p[i, :2], torch.tensor([1], device=device),
                                  p[i, 2:4], torch.tensor([1], device=device),
                                  p[i, 4:], torch.tensor([1], device=device)]).view(3, 3).T

        camera_points = torch.matmul(K_inv, pixel_points).T

        P_inv = torch.inverse(P)

        world_points_homogeneous = torch.matmul(P_inv, torch.cat((camera_points, torch.ones(3, 1, device=device)), dim=1).T).T
        world_points = world_points_homogeneous[:, :3] / world_points_homogeneous[:, 3].unsqueeze(1)

        normals.append(compute_normal_vector(world_points))

    normals_tensor = torch.stack(normals)
    return normals_tensor
