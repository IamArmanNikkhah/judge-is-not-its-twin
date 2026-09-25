#!/usr/bin/env python3
"""Native Colab client listing/release; no credentials are printed or copied."""
import argparse
import json
from colab_cli.auth import AuthProvider, get_credentials
from colab_cli.client import Client, Prod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-endpoint")
    args = ap.parse_args()
    client = Client(Prod(), get_credentials(provider=AuthProvider.OAUTH2))
    endpoints = {a.endpoint for a in client.list_assignments()}
    if args.stop_endpoint and args.stop_endpoint in endpoints:
        client.unassign(args.stop_endpoint)
        endpoints = {a.endpoint for a in client.list_assignments()}
    print(json.dumps({"endpoints": sorted(endpoints),
                      "target_absent": args.stop_endpoint not in endpoints if args.stop_endpoint else None}))


if __name__ == "__main__":
    main()
