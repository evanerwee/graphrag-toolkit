# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime

def get_properties_str(properties, default):
    if properties:
        return ';'.join(sorted([f'{k}:{v}' for k,v in properties.items()]))
    else:
        return default
        
def last_accessed_date(*args):
    return {
        'last_accessed_date': datetime.datetime.now().strftime("%Y-%m-%d")
    }