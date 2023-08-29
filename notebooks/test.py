import jax
import jaxlib
jax.config.update("jax_default_device", jax.devices()[1]

num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"jax version: {jax.__version__}")
print(f"jaxlib version: {jaxlib.__version__}")
print(f"Found {num_devices} JAX devices of type {device_type}.")

# jax.default_device = jax.devices("gpu")[1]
# jax.devices()
nmp = jax.numpy.ones(4)
print(nmp.device())
