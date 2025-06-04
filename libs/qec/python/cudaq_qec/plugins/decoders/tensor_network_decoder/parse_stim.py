# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

def convert_b8_to_bool(file_name: str, bits_per_shot: int) -> List[List[bool]]:
    with open(file_name, "rb") as f:
        data = f.read()
    return parse_b8(data, bits_per_shot)


def parse_b8(data: bytes, bits_per_shot: int) -> List[List[bool]]:
    """
    Convert a binary file to a list of lists of booleans.
    Copied from Stim's documentation.
    """
    shots = []
    bytes_per_shot = (bits_per_shot + 7) // 8
    for offset in range(0, len(data), bytes_per_shot):
        shot = []
        for k in range(bits_per_shot):
            byte = data[offset + k // 8]
            bit = (byte >> (k % 8)) % 2 == 1
            shot.append(bit)
        shots.append(shot)
    return shots