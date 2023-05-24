from absl import flags, app
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Main configuration file.")


def main(argv):
    """
    main The main function to execute the code.
    """
    print(FLAGS.config.experiment)


if __name__ == "__main__":
    app.run(main)
