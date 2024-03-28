from typing import List
import os
import shlex
class WSFSIntegration:
    def __init__(self, wsfs_path: str):
        """
        Class to represent the integration with Databricks WSFS.
        :params: wsfs_path: str Absolute path 
        """
        self.wsfs_path = os.path.normpath(wsfs_path)
        self.wsfs_dir = os.path.dirname(self.wsfs_path)
        
        # if not os.path.exists(self.wsfs_path):
        #     raise FileNotFoundError(f"The path {self.wsfs_path} does not exist.")
        
        
    def get_setup_command(self) -> List[str]:
        commands = [
            'export ZIP_OUTPUT="/tmp/repo.zip"',
            f'export WSFS_PATH="{self.wsfs_path}"',
            'export ENCODED_WSFS_PATH=$(python3 -c "import urllib.parse; print(urllib.parse.quote(\'$WSFS_PATH\'))")',
            f'mkdir -p "{self.wsfs_dir}"',
            'curl -X GET -o "$ZIP_OUTPUT" "$DATABRICKS_HOST/api/2.0/workspace/export?path=$ENCODED_WSFS_PATH&format=AUTO&direct_download=true" \
                -H "Authorization: Bearer $DATABRICKS_TOKEN"',
            f'''
            if file "$ZIP_OUTPUT" | grep -q "Zip archive data"; then
                apt update && apt install unzip
                unzip -d "{self.wsfs_dir}" "$ZIP_OUTPUT"
                rm -f "$ZIP_OUTPUT"
            else
                echo "$ZIP_OUTPUT is not a ZIP file."
            fi
            '''
        ]
        return commands
