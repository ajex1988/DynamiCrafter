import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default="A human face with natural expression and movement.", help="Prompt to use")
    parser.add_argument("-n", "--number", type=int, default=1, help="Repeat number")
    parser.add_argument("-o", "--out_file", type=str, default="test_prompts.txt", help="Repeat size")
    return parser.parse_args()


def gen_prompt_file(args):
    prompt = args.prompt
    number = args.number
    out_file = args.out_file
    with open(out_file, "w") as f:
        for i in range(number):
            if i == number-1:
                f.write(prompt)
            else:
                f.write(prompt + "\n")


def main():
    args = parse_args()
    gen_prompt_file(args)
    print("Done")


if __name__ == "__main__":
    main()
