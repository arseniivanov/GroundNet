import torch

def compute_normal_vector(world_points):
    vector1 = world_points[1] - world_points[0]
    vector2 = world_points[2] - world_points[0]
    normal = torch.cross(vector1, vector2, dim=0)
    normal_normalized = normal / torch.norm(normal)
    return normal_normalized

def transform_points_to_world(point_list, camera_data):
    batch_size = camera_data.shape[0]
    normals = []
    device = "cuda"

    for i in range(batch_size):
        P = camera_data[i, -16:].reshape(4, 4)
        
        # Initialize K with gradients disabled as it's a constant matrix
        K = torch.zeros((3, 3), requires_grad=False)
        K[0, 0] = camera_data[i, 16]
        K[1, 1] = camera_data[i, 17]
        K[0, 2] = camera_data[i, 18]
        K[1, 2] = camera_data[i, 19]
        K[2, 2] = 1.0

        K_inv = torch.inverse(K).to(device)

        # Using tensor operations to construct points
        x_points = point_list[0][i].unsqueeze(-1)  # Shape [3, 1]
        y_points = point_list[1][i].unsqueeze(-1)  # Shape [3, 1]
        ones = torch.ones(3, 1, device=device)
        pixel_points = torch.cat((x_points, y_points, ones), dim=1)  # Shape [3, 3]

        camera_points = torch.matmul(K_inv, pixel_points).T

        P_inv = torch.inverse(P).to(device)

        world_points_homogeneous = torch.matmul(P_inv, torch.cat((camera_points, torch.ones(3, 1).to(device)), dim=1).T).T
        world_points = world_points_homogeneous[:, :3] / world_points_homogeneous[:, 3].unsqueeze(1)

        normals.append(compute_normal_vector(world_points))

    normals_tensor = torch.stack(normals)
    return normals_tensor


def extract_points_from_heatmaps(heatmaps):
    # Assuming heatmaps shape is [batch_size, num_points, height, width]
    batch_size, num_points, height, width = heatmaps.size()

    # Create a coordinate grid
    x = torch.linspace(-1, 1, width).view(1, 1, 1, width).expand(batch_size, num_points, height, width).to("cuda")
    y = torch.linspace(-1, 1, height).view(1, 1, height, 1).expand(batch_size, num_points, height, width).to("cuda")

    # Apply softmax to heatmaps
    heatmaps = torch.softmax(heatmaps.view(batch_size, num_points, -1), dim=2).view_as(heatmaps)

    # Weighted sum of coordinates (soft-argmax)
    x = torch.sum(heatmaps * x, dim=[2, 3])
    y = torch.sum(heatmaps * y, dim=[2, 3])

    return x, y  # Each of shape [batch_size, num_points]
