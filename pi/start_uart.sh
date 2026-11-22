#!/bin/sh
# Switch on the header UART (pins 8/10) as /dev/ser1. Needed after every Pi reboot.
[ -e /dev/ser1 ] && { echo "/dev/ser1 already up"; exit 0; }
sudo msix-rp1 -m set 25l
sudo gpio-rp1 set 14,15 a4
sudo devc-serpl011-rpi5 -b115200 -v -c50000000 -e -F -u1 0x1f00030000,185 </dev/null >/dev/null 2>&1 &
sleep 2
sudo chmod 666 /dev/ser1 && echo "/dev/ser1 ready"
