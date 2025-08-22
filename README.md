# Spot Demo

![workflow](https://github.com/tomsilver/spot-planning-demo/actions/workflows/ci.yml/badge.svg)

## Real Spot Robot Setup Instructions

### Set Environment Variables

```bash
export BOSDYN_CLIENT_USERNAME=user
export BOSDYN_CLIENT_PASSWORD=<redacted>
export BOSDYN_IP="192.168.80.3"
```

### Create GraphNav Map

For now, we are still using the [Boston Dynamics SDK example script](https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/graph_nav_command_line/graph_nav_util.py). Later we should make this our own.

Save the downloaded map in `graph_nav_maps/` with a distinctive name.

### Join Spot's Wifi Network

Just look for a Wifi network the starts with "spot-" and join from your computer.

