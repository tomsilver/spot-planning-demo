# Spot Demo

![workflow](https://github.com/tomsilver/spot-planning-demo/actions/workflows/ci.yml/badge.svg)

## Real Spot Robot Setup Instructions

### Hardware Notes

To use perception or any APIs over Wifi (LLMs etc.), we need to be able to connect to both the robot and the internet at the same time. The main way that we do this is by connecting to the robot over Wifi, and connecting to the internet over ethernet (from the laptop/desktop). So before you get started with perception, you will need an ethernet connection. At Princeton, this may require registering your device through IT in addition to purchasing the appropriate cables and adapters. A temporary workaround (not recommended for long, because of cellular data costs and latency) is to tether your phone via USB, instead of using ethernet. Ask ChatGPT for instructions on that.

### APIs

For now, we are using Gemini for perception. Make sure you do:
```bash
export GOOGLE_API_KEY=<redacted>
```

### Set Environment Variables

```bash
export BOSDYN_CLIENT_USERNAME=user
export BOSDYN_CLIENT_PASSWORD=<redacted>
export BOSDYN_IP="192.168.80.3"
```

### Create GraphNav Map

**Do this once per real-world environment.**

For now, we are still using the [Boston Dynamics SDK example script](https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/graph_nav_command_line/graph_nav_util.py). Later we should make this our own.

Save the downloaded map in `graph_nav_maps/` with a distinctive name.

### Join Spot's Wifi Network

Just look for a Wifi network the starts with "spot-" and join from your computer.

