sed -e 's/Jan/01/1' -e 's/Feb/02/1' -e 's/Mar/03/1' -e 's/Apr/04/1' -e 's/May/05/1' -e 's/Jun/06/1' -e 's/Jul/07/1' -e 's/Aug/08/1' -e 's/Sep/09/1' -e 's/Oct/10/1' -e 's/Nov/11/1' -e 's/Dec/12/1' -e 's/ /-/1' -e 's/^.*/2021-&/1' -e 's/- /-0/1' | 
sed -e 's/^\([^a-zA-Z]*\).*sudo.*:[ ]*\([^ ]*\) :.*COMMAND=\(.*\)/\2 used sudo to do `\3` on \1/' -e '/used sudo/w audit_sudo.txt' | 
sed -e 's/^.*PAM: Auth.*for \([^ ]*\) from \(.*\)/usr=\1@ip=\2/' -e 's/^.*message repeated //' -e 's/^.*PAM:.*illegal.*from \(.*\)/ip=\1/' | 
sed -e '/.*[\.].*/h' -e '/2 times/G' -e '/2 times/G' -e '/1 times/G' | 
sed -r -e '/usr=|ip=/!d' | 
tr '@' '\n' | 
sort | 
uniq -c | 
sed 's/[ ]*\([0-9]*\) \(.*\)/\2 failed to log in \1 times/' | 
sed -e 's/ip=//' -e '/usr/!w audit_ip.txt' -e '/usr/!d' | 
sed -n -e 's/usr=//' -e '/^.*/w audit_user.txt'
