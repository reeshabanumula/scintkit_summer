def pair_receiver_files(receiverA_files, receiverB_files, cf): #takes in configuration file as a parameter

    receiverB_dict = {
        extract_file_datetime(file): file
        for file in receiverB_files
    }

    paired_files = []

    for fileA in receiverA_files:

        datetimeA = extract_file_datetime(fileA)

        # ---------------- EXACT ----------------

        if cf.pairing_mode == "exact":

            if datetimeA in receiverB_dict:
                paired_files.append((fileA, receiverB_dict[datetimeA]))

            elif cf.unpaired_file_action == "skip":
                print(f"Skipping {fileA.name}")

            else:
                raise ValueError(f"No matching file for {fileA.name}")

        # ---------------- NEAREST ----------------

        elif cf.pairing_mode == "nearest":

            closest = min(receiverB_dict.keys(), key=lambda t: abs((t - datetimeA).total_seconds()))

            difference = abs((closest - datetimeA).total_seconds()) / 60

            if difference <= cf.pairing_tolerance:
                paired_files.append((fileA, receiverB_dict[closest]))

            elif cf.unpaired_file_action == "skip":
                print(f"Skipping {fileA.name}")

            else:
                raise ValueError(f"No nearby match for {fileA.name}")

        else:
            raise ValueError("Invalid pairing_mode")

    return paired_files