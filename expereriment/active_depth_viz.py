import torch
import matplotlib.pyplot as plt
from matplotlib import cm

import depth_cov.core.gaussian_kernel as gk
from depth_cov.core.NonstationaryGpModule import NonstationaryGpModule
from depth_cov.data.depth_transforms import BaseTransform
from depth_cov.utils.utils import normalize_coordinates, sample_coords, to_pixel_covariance
from depth_cov.utils.viz_utils import remove_all_ticks

from PIL import Image
import numpy as np
import cv2
import torchvision.transforms.functional as TF

cmap = "viridis"

# ScanNet loader
def get_rgb_depth(rgb_filename, depth_filename, size):
  # rgb_pil = Image.open(rgb_filename)
  # rgb = TF.to_tensor(rgb_pil)
  # depth_np = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
  # depth_np = depth_np.astype(np.float32) / 1000.0
  # depth = torch.from_numpy(depth_np)
  # depth = torch.unsqueeze(depth, dim=0)
  # depth[depth<=0.0] = float('nan')
  # log_depth = torch.log(depth)

  # transform = BaseTransform(size)
  # rgb, log_depth = transform(rgb, log_depth)

  # return rgb, log_depth  

  # =================================================================================
  # rgb_pil = Image.open(rgb_filename)
  # rgb = TF.to_tensor(rgb_pil)
  # depth_np = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
  # depth_np = depth_np.astype(np.float32) / 1000.0
  # depth = torch.from_numpy(depth_np)
  # depth = torch.unsqueeze(depth, dim=0)
  # depth[depth <= 0.0] = float('nan')
  # log_depth = torch.log(depth)
  # # Calculate 100 * max(R, G, B) / 255 for each pixel
  # max_rgb = torch.max(rgb, dim=0)[0]  # get maximum value along the channel dimension
  # max_rgb_percent = 100.0 * max_rgb / 255.0  # calculate percentage
  # max_rgb_percent = max_rgb_percent.unsqueeze(0)  # add batch dimension
  # # Concatenate log_depth and max_rgb_percent along the channel dimension
  # depth_with_max_rgb_percent = torch.cat((log_depth, max_rgb_percent), dim=0)
  # transform = BaseTransform(size)
  # rgb, depth_with_max_rgb_percent = transform(rgb, depth_with_max_rgb_percent)
  # return rgb, depth_with_max_rgb_percent

  # ===================================================================================
  # Load and transform the RGB image
  rgb_pil = Image.open(rgb_filename)
  rgb = TF.to_tensor(rgb_pil)
  
  # Calculate the max RGB value and normalize it
  max_rgb = torch.max(rgb, dim=0)[0]  # Take max value along the RGB channels
  normalized_max_rgb = 100 * max_rgb / 255.0
  
  # Load and transform the depth image
  depth_np = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
  depth_np = depth_np.astype(np.float32) / 1000.0
  depth = torch.from_numpy(depth_np)
  depth = torch.unsqueeze(depth, dim=0) * normalized_max_rgb  # Adjust depth values by normalized max RGB
  depth[depth <= 0.0] = float('nan')  # Set non-positive depth values to NaN
  
  # Calculate the logarithm of the depth
  log_depth = torch.log(depth)
  
  # Apply transformations
  transform = BaseTransform(size)
  rgb, log_depth = transform(rgb, log_depth)
  return rgb, log_depth


def get_data(rgb_path, depth_path, num_samples, size, device):
  # rgb, gt_depth_with_max_rgb_percent = get_rgb_depth(rgb_path, depth_path, size)
  # rgb = rgb.unsqueeze(0)
  # gt_depth_with_max_rgb_percent = gt_depth_with_max_rgb_percent.unsqueeze(0)

  # coords_train, depth_train, batch_train = sample_coords(gt_depth_with_max_rgb_percent, None, num_samples, mode="uniform")

  # # Setup test coords
  # h = rgb.shape[-2]
  # w = rgb.shape[-1]
  # y_coords, x_coords = torch.meshgrid(torch.arange(h, device='cpu'), torch.arange(w, device='cpu'), indexing='ij')
  # test_coords = torch.column_stack((torch.flatten(y_coords), torch.flatten(x_coords)))

  # rgb = rgb.to(device)
  # gt_depth_with_max_rgb_percent = gt_depth_with_max_rgb_percent.to(device)
  # coords_train = coords_train.to(device)
  # depth_train = depth_train.to(device)
  # test_coords = test_coords.to(device)

  # return rgb, gt_depth_with_max_rgb_percent, coords_train, depth_train, test_coords

  # =======================================================================================

  # rgb, gt_depth_with_max_rgb_percent_with_max_rgb_percent = get_rgb_depth(rgb_path, depth_path, size)

  # coord_train, depth_train, batch_train = sample_coords(gt_depth_with_max_rgb_percent_with_max_rgb_percent[:1, ...], None, num_samples, mode="uniform")
  
  # print("gt_depth_with_max_rgb_percent_with_max_rgb_percent shape:", gt_depth_with_max_rgb_percent_with_max_rgb_percent.shape)
  # print("coord_train shape:", coord_train.shape)
  # print("depth_train shape:", depth_train.shape)
  
  # # max_rgb_percent_train = gt_depth_with_max_rgb_percent_with_max_rgb_percent[1:, ...].expand_as(depth_train)
  # max_rgb_percent = gt_depth_with_max_rgb_percent_with_max_rgb_percent[1:, ...]
  # max_rgb_percent_train = max_rgb_percent[0, coord_train[0, :, 0], coord_train[0, :, 1]].unsqueeze(0).unsqueeze(2)

  # rgb = rgb.unsqueeze(0)
  # gt_depth_with_max_rgb_percent = gt_depth_with_max_rgb_percent_with_max_rgb_percent[:1, ...]  # extract original depth data
  # rgb = rgb.to(device)
  # gt_depth_with_max_rgb_percent = gt_depth_with_max_rgb_percent.to(device)
  # coord_train = coord_train.to(device)
  # # depth_train_with_max_rgb_percent = torch.cat((depth_train, max_rgb_percent_train), dim=0).to(device)
  # depth_train_with_max_rgb_percent = torch.cat((depth_train, max_rgb_percent_train), dim=2).to(device)

  # h = rgb.shape[-2]
  # w = rgb.shape[-1]
  # y_coords, x_coords = torch.meshgrid(torch.arange(h, device='cpu'), torch.arange(w, device='cpu'), indexing='ij')
  # test_coords = torch.column_stack((torch.flatten(y_coords), torch.flatten(x_coords))).to(device)
  # return rgb, gt_depth_with_max_rgb_percent, coord_train, depth_train_with_max_rgb_percent, test_coords

  # =========================================================================================
  rgb, gt_depth_with_max_rgb_percent = get_rgb_depth(rgb_path, depth_path, size)
  rgb = rgb.unsqueeze(0)
  gt_depth_with_max_rgb_percent = gt_depth_with_max_rgb_percent.unsqueeze(0)

  coord_train, depth_train, batch_train = sample_coords(gt_depth_with_max_rgb_percent, None, num_samples, mode="uniform")

  rgb = rgb.to(device)
  gt_depth_with_max_rgb_percent = gt_depth_with_max_rgb_percent.to(device)
  coord_train = coord_train.to(device)
  depth_train = depth_train.to(device)

  h = rgb.shape[-2]
  w = rgb.shape[-1]
  y_coords, x_coords = torch.meshgrid(torch.arange(h, device='cpu'), torch.arange(w, device='cpu'), indexing='ij')
  test_coords = torch.column_stack((torch.flatten(y_coords), torch.flatten(x_coords)))
  test_coords = test_coords.to(device)
  return rgb, gt_depth_with_max_rgb_percent, coord_train, depth_train, test_coords

def plot_rgb_and_coords(ax, img, coords):
  batch_ind = 0
  ax.clear()
  ax.imshow(torch.permute(img[batch_ind,...], (1,2,0)))
  ax.scatter(x = coords[batch_ind,:,1].cpu(), y = coords[batch_ind,:,0].cpu(), c='r', s=10)
  ax.title.set_text("RGB and " "{:3d}".format(coords.shape[1]) + " Samples")


def select_pair(axs, row, col):
  # Plot two Gaussian kernels on rgb image

  x1, y1 = [-0.5, K.shape[-1]-0.5], [row-0.5, row-0.5]
  x2, y2 = [-0.5, K.shape[-1]-0.5], [row+0.5, row+0.5]
  x3, y3 = [col-0.5, col-0.5], [-0.5, K.shape[-2]-0.5]
  x4, y4 = [col+0.5, col+0.5], [-0.5, K.shape[-2]-0.5]

  x_lim_before = axs[1][0].get_xlim()
  y_lim_before = axs[1][0].get_ylim()
  axs[1][0].clear()
  axs[1][0].imshow(K[0,:,:].detach(), cmap=cmap)
  axs[1][0].plot(x1, y1, x2, y2, x3, y3, x4, y4, color='r', linewidth=1)
  axs[1][0].set_xlim(x_lim_before)
  axs[1][0].set_ylim(y_lim_before)
  axs[1][0].title.set_text("Training Covariance Matrix")

  kernel_mu1 = coords_train[0:1,row:row+1,:]
  kernel_mu1_norm = normalize_coordinates(kernel_mu1, rgb.shape[-2:])
  kernel_mu2 = coords_train[0:1,col:col+1,:]
  kernel_mu2_norm = normalize_coordinates(kernel_mu2, rgb.shape[-2:])

  # axs[0][1].clear()
  plot_rgb_and_coords(axs[0][0], rgb, coords_train)
  K1_map = model.get_correlation_map(gaussian_covs, -1, kernel_mu1_norm, rgb.shape[-2:])
  axs[1][1].imshow(K1_map[0,...], cmap=cmap)
  axs[1][1].title.set_text("Correlation Map " "{:3d}".format(row))
  K1_map = model.get_correlation_map(gaussian_covs, -1, kernel_mu2_norm, rgb.shape[-2:])
  axs[1][2].imshow(K1_map[0,...], cmap=cmap)
  axs[1][2].title.set_text("Correlation Map " "{:3d}".format(col))

def on_click(event, axs):

  global rgb
  global gt_depth_with_max_rgb_percent
  global coords_train
  global depth_train

  global model
  global gaussian_covs
  global mean_depth
  global K

  # RGB
  if event.inaxes == axs[0][0] or event.inaxes == axs[0][2]:
    row = round(event.ydata)
    col = round(event.xdata)

    # Add new sparse coords and depth point and update scatter plot
    new_coord = torch.tensor([[[row, col]]])
    new_depth = torch.tensor([[[gt_depth_with_max_rgb_percent[0, 0, row, col]]]])

    if ~new_depth.isnan():
      coords_train = torch.cat((coords_train, new_coord), dim=1)
      depth_train = torch.cat((depth_train, new_depth), dim=1)
      plot_rgb_and_coords(axs[0][0], rgb, coords_train)

      coords_train_norm = normalize_coordinates(coords_train, rgb.shape[-2:])
      pred_depths, pred_vars = model.condition(gaussian_covs, coords_train_norm, depth_train, mean_depth, test_size)

      mask = ~gt_depth_with_max_rgb_percent.isnan()
      vmin = torch.min(gt_depth_with_max_rgb_percent[mask])
      vmax = torch.max(gt_depth_with_max_rgb_percent[mask])

      pred_depth_viz = pred_depths[-1][0,0,:,:].cpu().detach()
      axs[0][1].imshow(pred_depth_viz, vmin=vmin, vmax=vmax)
      axs[0][1].title.set_text("Conditional Mean")
      pred_vars_viz = pred_vars[-1][0,0,:,:].cpu().detach()
      pred_std_viz = torch.sqrt(pred_vars_viz)
      axs[0][2].imshow(pred_std_viz)
      axs[0][2].title.set_text("Conditional St Dev")


      K, _ = model.get_covariance(gaussian_covs, -1, coords_train_norm)
      axs[1][0].imshow(K[0,:,:].detach(), cmap=cmap)
      axs[1][0].relim()
      axs[1][0].autoscale()
      axs[1][0].title.set_text("Training Covariance Matrix")
    else:
      print("Selected invalid depth!")

  # Kernel matrix
  if event.inaxes == axs[1][0]:
    # Plot two Gaussian kernels on rgb image
    row = round(event.ydata)
    col = round(event.xdata)
    select_pair(axs, row, col)

  plt.draw()


def main(model_path, rgb_path, depth_path, num_samples_init, test_size, device):

  global rgb
  global gt_depth_with_max_rgb_percent
  global coords_train
  global depth_train

  global model
  global gaussian_covs
  global mean_depth
  global K
  
  # Get data
  size = torch.Size([192, 256])
  rgb, gt_depth_with_max_rgb_percent, coords_train, depth_train, test_coords = get_data(rgb_path, depth_path, num_samples_init, size, device)
  mean_depth = torch.nanmean(gt_depth_with_max_rgb_percent)

  # Eval model
  model = NonstationaryGpModule.load_from_checkpoint(model_path, train_size=test_size)
  model.eval()
  model.to(device)
  # Run network
  gaussian_covs = model(rgb)

  # Visualization setup
  fig, axs = plt.subplots(2,3,figsize=(10,6))
  remove_all_ticks(axs)

  # RGB and sparse points
  plot_rgb_and_coords(axs[0][0], rgb, coords_train)

  # Predicted depth
  coords_train_norm = normalize_coordinates(coords_train, rgb.shape[-2:])
  pred_depths, pred_vars = model.condition(gaussian_covs, coords_train_norm, depth_train, mean_depth, test_size)

  mask = ~gt_depth_with_max_rgb_percent.isnan()
  vmin = torch.min(gt_depth_with_max_rgb_percent[mask])
  vmax = torch.max(gt_depth_with_max_rgb_percent[mask])

  pred_depth_viz = pred_depths[-1][0,0,:,:].cpu().detach()
  axs[0][1].imshow(pred_depth_viz, vmin=vmin, vmax=vmax)
  axs[0][1].title.set_text("Conditional Mean")

  pred_vars_viz = pred_vars[-1][0,0,:,:].cpu().detach()
  pred_std_viz = torch.sqrt(pred_vars_viz)
  axs[0][2].imshow(pred_std_viz)
  axs[0][2].title.set_text("Conditional St Dev")

  K, _ = model.get_covariance(gaussian_covs, -1, coords_train_norm)
  axs[1][0].imshow(K[0,:,:].detach(), cmap=cmap)
  axs[1][0].title.set_text("Training Covariance Matrix")

  select_pair(axs, row=1, col=0)

  cid = fig.canvas.mpl_connect('button_press_event', lambda x: on_click(x, axs))
  plt.show()
  plt.savefig("active_300.png")

if __name__ == "__main__":
  # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu")

  # model_path = "models/scannet.ckpt"

  #base
  model_path = "/root/nrkim/DepthCov/depth_cov/models/real_300_epochs_results-loss_val=-61.9568.ckpt"

  # our_model
  # model_path = "/root/dhkim/DepthCov/depth_cov/models/gp-epoch=288-loss_val=-62.0949.ckpt"

  rgb_path = "../dataset/examples/frame-00000.color.jpg"
  depth_path = "../dataset/examples/frame-00000.depth.pgm"

  test_size = torch.Size([192, 256])

  num_samples_init = 5

  with torch.no_grad():
    main(model_path, rgb_path, depth_path, num_samples_init, test_size, device)