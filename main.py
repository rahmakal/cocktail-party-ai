import importlib
import sys


COMMANDS = {
    "generate": "src.generate_mixtures",
    "train": "src.train",
    "evaluate": "src.evaluate",
    "separate": "src.separate",
    "arabic-mixtures": "src.mixture",
}


def print_usage():
    print("Usage: python main.py <command> [options]\n")
    print("Commands:")
    for command in COMMANDS:
        print(f"  {command}")
    print("\nExamples:")
    print("  python main.py generate --src_dir data/raw_sources --n_mix 8000 --n_src 3")
    print("  python main.py train")
    print("  python main.py evaluate --ckpt checkpoints_3src_2/best.ckpt")
    print("  python main.py separate --mix path/to/mixture.wav --ckpt checkpoints_3src_2/best.ckpt")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print_usage()
        return 0

    command = sys.argv[1]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown command: {command}\n")
        print_usage()
        return 2

    module = importlib.import_module(module_name)
    sys.argv = [f"main.py {command}", *sys.argv[2:]]
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
