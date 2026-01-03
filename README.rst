qwack
=====

qwack: a quickly written hack_ (1985) variant for Python.

.. _hack: https://en.wikipedia.org/wiki/Hack_%28Unix_video_game%29



Keyboard Controls
-----------------

Movement (vi-style):

- ``h`` West
- ``j`` South
- ``k`` North
- ``l`` East
- ``y`` Northwest
- ``u`` Northeast
- ``b`` Southwest
- ``n`` Southeast

Actions (ultima 4 style):

- ``o`` Open door
- ``t`` Talk to NPC
- ``E`` Enter portal/city
- ``K`` Klimb ladder up
- ``D`` Descend ladder down
- ``B`` Board ship/mount horse
- ``X`` eXit ship/unmount horse
- ``C`` Cast spell
    - Only one spell available, press ``C`` again for confusion
    - requires 'chafa' to be installed


Wizard Mode, enabled by default (``Ctrl-W`` to toggle):

- ``1`` Toggle clipping
- ``A`` Auto resize tiles
- ``R`` Toggle radius
- ``[`` / ``]`` Adjust darkness
- ``(`` / ``)`` Adjust radius
- ``{`` / ``}`` Adjust tile size
- ``<`` / ``>`` Adjust char size
- ``Ctrl-R`` Cycle tileset
- ``Ctrl-T`` Cycle charset
- ``Ctrl-D`` Toggle debug display


Changing tile sizes
-------------------

Only the default tile size (16) and character size (8) are shipped with the game
through pypi. These tiles are pre-generated, and so an external binary (chafa)
is not required.

They don't have to be, though, as long as 'chafa' is installed to
/usr/local/bin/chafa or the path specified by environment variable
``CHAFA_BIN``, you can then use any of the wizard keys for cycling tileset and
charset and adjusting tile and char size.
