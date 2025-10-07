#!/usr/bin/env python3

import os
import sys

def fix_import():
    """
    Fix the import issue in deeprobust's env.py file.
    
    The error is: ModuleNotFoundError: No module named 'scipy.sparse.linalg.eigen.arpack'
    This happens because in newer versions of SciPy, arpack is directly under scipy.sparse.linalg
    """
    # Get the path to the env.py file
    import site
    for site_path in site.getsitepackages():
        env_path = os.path.join(site_path, 'deeprobust/graph/rl/env.py')
        if os.path.exists(env_path):
            print(f"Found env.py at: {env_path}")
            
            # Read the file
            with open(env_path, 'r') as f:
                content = f.read()
            
            # Replace the import
            old_import = "from scipy.sparse.linalg.eigen.arpack import eigsh"
            new_import = "from scipy.sparse.linalg import eigsh"
            
            if old_import in content:
                content = content.replace(old_import, new_import)
                
                # Backup the original file
                backup_path = env_path + '.bak'
                print(f"Creating backup at: {backup_path}")
                with open(backup_path, 'w') as f:
                    f.write(content)
                
                # Write the fixed content
                with open(env_path, 'w') as f:
                    f.write(content)
                
                print(f"Fixed import in {env_path}")
                return True
            else:
                print(f"Could not find the import line in {env_path}")
    
    print("Could not find the env.py file")
    return False

if __name__ == "__main__":
    success = fix_import()
    if success:
        print("Import fixed successfully. Try running the command again.")
    else:
        print("Failed to fix the import. You might need to manually fix it or update SciPy.") 