import os


def find_python_files(directory):
    python_files = []
    for file in os.listdir(directory):
        if file.endswith(".py"):
            python_files.append(os.path.join(directory, file))
    return python_files


def create_bash_script(python_files, bash_file):
    python_files = [os.path.splitext(os.path.basename(path))[0] for path in python_files]
    if '__init__' in python_files:
        python_files.remove('__init__')
    with open(bash_file, "w") as f:
        # f.write("#!/bin/bash\n")
        for file in python_files:
            f.write(f"python cliport/demos.py n=1 task={file} mode=val record.save_video=True\n")


def main():
    directory = r'./cliport/generated_tasks/'
    bash_file = 'run.bat'

    if os.path.exists(directory):
        python_files = find_python_files(directory)
        if python_files:
            print("Python files found in the root directory:")
            for file in python_files:
                print(file)
            create_bash_script(python_files, bash_file)
            print(f"Bash file '{bash_file}' created successfully with Python files.")
        else:
            print("No Python files found in the root directory.")
    else:
        print("Directory does not exist.")


if __name__ == "__main__":
    main()
