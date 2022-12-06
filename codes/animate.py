import os, sys
import numpy as np
import json
import ffmpeg

def find_files(find_str, pwd_path):
    message = "searching for " + find_str + ": "
    sys.stdout.write(message)
    filenames = []
    for filename in os.popen("find " + str(pwd_path) + " -path "
                            + '"' + str(find_str) + '"').read().split('\n')[0:-1]:
        filenames.append(filename)
    message = str(len(filenames)) + " files found.\n"
    sys.stdout.write(message)
    return filenames

path_in = "."
fns = np.asarray(find_files('*lensing_device_eom.jpg*', f'./{path_in}'))

p01s = []
fds = []
alphas = []
for fn in fns:
    json_fn = fn.replace('lensing_device_eom.jpg','params.json')
    with open(json_fn) as file:
        params = json.load(file)
    fds.append(params['fd'])
    p01s.append(params['pos0'][1])
    alphas.append(params['alpha'])
p01s = np.asarray(p01s)
alphas = np.asarray(alphas)
fds = np.asarray(fds)

# order = np.argsort(alphas)
# p01s = np.take_along_axis(p01s, order, axis=0)
# alphas = np.take_along_axis(alphas, order, axis=0)
# fns = np.flip(np.take_along_axis(fns, order, axis=0))

fds_unique = np.unique(fds)
print(fds_unique[[0,2]])  # select the relevant ones
for fd in fds_unique[[0,2]]:
    fns_slice = fns[fds==fd]
    p01s_slice = p01s[fds==fd]
    alphas_slice = alphas[fds==fd]
    for a in np.unique(alphas_slice):
        fns_slice = fns_slice[alphas_slice==a]
        p01s_slice = p01s_slice[alphas_slice==a]
        order = np.argsort(p01s_slice)
        fns_slice = np.flip(np.take_along_axis(fns_slice, order, axis=0))

        # List of JPEG files
        jpeg_files = fns_slice
        print(jpeg_files)
        # continue
        # exit()

        # Execute FFmpeg sub-process, with stdin pipe as input, and jpeg_pipe input format
        process = ffmpeg.input('pipe:', r=1).output(f'/tmp/video_{a}_{fd}.mp4', vcodec='libx264').overwrite_output().run_async(pipe_stdin=True)

        # Iterate jpeg_files, read the content of each file and write it to stdin
        for in_file in jpeg_files:
            with open(in_file, 'rb') as f:
                # Read the JPEG file content to jpeg_data (bytes array)
                jpeg_data = f.read()

                # Write JPEG data to stdin pipe of FFmpeg process
                process.stdin.write(jpeg_data)

        # Close stdin pipe - FFmpeg fininsh encoding the output file.
        process.stdin.close()
        process.wait()