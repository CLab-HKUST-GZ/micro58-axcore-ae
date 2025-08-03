#!/bin/bash
cd ./synth
find . -depth -mindepth 1 -not -path './dc_syn.tcl' -delete

echo -n "DC synthesis is running, please wait ...  "  # Use -n to keep the cursor on the same line


# 1. Run your main command in the background (&) and hide all its output.
dc_shell -f dc_syn.tcl > /dev/null 2>&1 &

# 2. Get the Process ID (PID) of the background task.
PID=$!

# 3. Define the characters for a spinning animation.
spinner="/-\|"

# 4. While the background process is still running, loop and display the animation.
while ps -p $PID > /dev/null; do
    # Use \b (backspace) to overwrite the previous character, 
    # creating an in-place spinning effect.
    printf "\b%s" "${spinner:i++%4:1}"
    sleep 0.1
done

# 5. The background task is finished, print a completion message.
#    \b erases the spinner character.
printf "\bDone!\n"

# Optional: Check the exit status of the original command.
# 'wait $PID' waits for the background process to end and returns its exit code.
wait $PID
if [ $? -eq 0 ]; then
    echo "Status: DC synthesis is finished."
else
    echo "Status: An error may have occurred during execution."
fi