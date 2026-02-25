"""Gateways package."""
from swarm.gateways.base import BaseGateway
from swarm.gateways.cli_gateway import CLIGateway
from swarm.gateways.web_gateway import WebGateway
from swarm.gateways.email_gateway import EmailGateway

__all__ = ["BaseGateway", "CLIGateway", "WebGateway", "EmailGateway"]
