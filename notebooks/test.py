import jax

jax.default_device = jax.devices("gpu")[1]
jax.devices()
nmp = jax.numpy.ones(4)
print(nmp.device())
