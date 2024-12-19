import os
import json
import re


def browse_py_files(folder):
    """
    Browse all the .py files in the specified folder.
    """
    py_files = []
    for file in os.listdir(folder):
        if file.endswith('.py'):
            py_files.append(file)
        if '__init__.py' in py_files:
            py_files.remove('__init__.py')
    return py_files


def get_urdfs_and_description(py_file):
    """
    Extract URDFs and task description from the Python file.
    """
    urdfs = []
    description = ''
    with open(py_file, 'r') as f:
        lines = f.readlines()
        in_method = False
        for line in lines:
            # Extract task description
            if 'class' in line and description == '':
                # Look for a comment right under the class definition
                if lines[lines.index(line) + 1].strip().startswith('"""'):
                    for next_line in lines[lines.index(line) + 1:]:
                        if lines.index(next_line) > lines.index(line) + 1:
                            description += ' '
                        description += next_line.strip()
                        if next_line.strip().endswith('"""'):
                            description = description[3:-3]
                            break
                else:
                    # If no comment found, refer to self.lang_template in __init__() method
                    for next_line in lines[lines.index(line) + 1:]:
                        if 'self.lang_template' in next_line:
                            description = next_line.split('=')[-1].strip().strip('"')
                            break

            # Extract URDFs
            # Search for method definition
            if 'def reset(' in line:
                in_method = True

            # Extract URDFs within method body
            if in_method:
                urdf_matches = re.findall(r"'(.*?\.urdf)'", line)
                urdfs.extend(urdf_matches)

    return urdfs, description


def update_generated_tasks(py_files, json_file):
    """
    Update the generated_tasks.json file with information from Python files.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Convert the list to a set for faster lookup
    registered_files = set(data.keys())

    for py_file in py_files:
        # Convert '_' to '-' and remove '.py' extension for key
        key = py_file.replace('_', '-').replace('.py', '')

        if key not in registered_files:
            urdfs, description = get_urdfs_and_description(
                os.path.join(r'./cliport/generated_tasks', py_file))

            # Format URDFs list
            # formatted_urdfs = []
            # for urdf in urdfs:
            #    formatted_urdfs.append(f'{urdf}')

            # Update JSON data
            data[key] = {
                "task-name": key,
                "task-description": description,
                "assets-used": urdfs
            }

    # Write updated data back to JSON file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def update_json(file_list, json_file):
    """
    Update the JSON file with the list of Python files.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Convert the list to a set for faster lookup
    registered_files = set(data)

    # Check for missing files
    missing_files = [file for file in file_list if file not in registered_files]

    # Update JSON data
    if missing_files:
        data.extend(missing_files)
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)


def main():
    # Folder containing Python files
    folder = r'./cliport/generated_tasks'
    # JSON file containing registered file names
    json_file = r'./prompts/data/generated_task_codes.json'
    generated_tasks_json = r'./prompts/data/generated_tasks.json'

    # Browse .py files
    py_files = browse_py_files(folder)

    # Update JSON file
    update_json(py_files, json_file)

    # Update generated_tasks.json file
    update_generated_tasks(py_files, generated_tasks_json)

    print("Updated JSON file with missing Python files.")


if __name__ == "__main__":
    main()
