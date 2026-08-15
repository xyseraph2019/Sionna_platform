"""
Deprecated: component protocols have been folded into the concrete modules.

This module is kept only for backward compatibility. New code should not import
from here; built-in channels/receivers are now constructed directly in
``sionna5g.channel`` and ``sionna5g.receiver``. Custom components still use the
small registry in ``sionna5g.registry``.
"""
