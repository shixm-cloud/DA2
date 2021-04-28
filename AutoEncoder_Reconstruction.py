# Apply Field Reconstruction to Obs from Nature Run

# Import libraries

import numpy as np
import netCDF4 as nc4
import tensorflow as tf


# Load the model dataset, scale, and mask

dsFile = nc4.Dataset("nature_barotropic.nc")
streamNat = np.copy(dsFile.variables["stream"]).astype("float32")
vorNat = np.copy(dsFile.variables["vor"]).astype("float32")
vorNatSH = -np.copy(vorNat[:, 0:49, :])
vorNat[:, 0:49, :] = np.copy(vorNatSH)
streamStd = 6944374.000000
streamAve = -1021447.5625000
psi0 = (streamNat - streamAve) / streamStd
psi0masked = np.where(vorNat > 5e-6, -100.0, psi0)
psi0masked[:, 0:49, :] = np.where(vorNat[:, 0:49, :] > 5e-6, 100.0, psi0[:, 0:49, :])

obs_images = psi0masked.reshape(
    (streamNat.shape[0], streamNat.shape[1], streamNat.shape[2], 1)
).astype("float32")
rec_images = np.empty(
    [streamNat.shape[0], streamNat.shape[1], streamNat.shape[2], 1]
).astype("float32")


# Set up DL dataset
batch_size = 64
obs_dataset = tf.data.Dataset.from_tensor_slices(obs_images).batch(batch_size)

# Load the encoder and decoder networks
autoEncoder = tf.keras.models.load_model("./StreamAutoEncoder_L1024")
autoEncoder.compile()

# Reconstruct
ts = 0
te = batch_size
for obs_x in obs_dataset:
    rec_images[ts:te, :, :, :] = autoEncoder(obs_x)
    ts = ts + batch_size
    te = te + batch_size
    if te > streamNat.shape[0] + 1:
        te = streamNat.shape[0] + 1


# save reconstructed dataset for visualization
ilat = np.copy(dsFile.variables["lat"])
ilon = np.copy(dsFile.variables["lon"])
ncfile = nc4.Dataset(
    "./reconstruct_barotropic_l1024.nc", mode="w", format="NETCDF4_CLASSIC"
)
lat_dim = ncfile.createDimension("lat", ilat.shape[0])
lon_dim = ncfile.createDimension("lon", ilon.shape[0])
time_dim = ncfile.createDimension("time", None)

lat = ncfile.createVariable("lat", np.float32, ("lat",))
lat.units = "degrees_north"
lat.long_name = "latitude"
lon = ncfile.createVariable("lon", np.float32, ("lon",))
lon.units = "degrees_east"
lon.long_name = "longitude"
time = ncfile.createVariable("time", np.float32, ("time",))
time.units = "days since beginning"
time.long_name = "time"

nat = ncfile.createVariable(
    "streamNat", np.float32, ("time", "lat", "lon"), fill_value=1.0e36
)
nat.units = "none"
nat.standard_name = "streamfunction"
obs = ncfile.createVariable(
    "streamObs", np.float32, ("time", "lat", "lon"), fill_value=1.0e36
)
obs.units = "none"
obs.standard_name = "streamfunction"
rec = ncfile.createVariable(
    "streamRec", np.float32, ("time", "lat", "lon"), fill_value=1.0e36
)
rec.units = "none"
rec.standard_name = "streamfunction"
mix = ncfile.createVariable(
    "streamMix", np.float32, ("time", "lat", "lon"), fill_value=1.0e36
)
mix.units = "none"
mix.standard_name = "streamfunction"

lat[:] = ilat
lon[:] = ilon
nat[:, :, :] = streamNat[:, :, :]
obs_images = np.where(np.abs(obs_images) > 99.0, np.nan, obs_images)
obs_images = obs_images * streamStd + streamAve
obs[:, :, :] = obs_images[:, :, :, 0]
rec_images = rec_images * streamStd + streamAve
rec[:, :, :] = rec_images[:, :, :, 0]
mix_images = np.copy(obs_images)
mix_images = np.where(np.isnan(obs_images), rec_images, mix_images)
mix[:, :, :] = mix_images[:, :, :, 0]
time[:] = np.arange(0, streamNat.shape[0])

ncfile.close()
