#!/bin/sh
OK=0
CANCEL=1
ALL=3
ESC=255

EntrancePage(){

    Option=$(dialog --title "System Info Panel" --clear \
	--menu "Please select the command you want to use" 20 50 8 \
	1 "POST ANNOUNCEMENT" \
	2 "USER LIST" \
    	2>&1 > /dev/tty)

    ExitCode=$?
    if [ $ExitCode -eq $CANCEL ]; then
	Msg=$(echo "Exit.")
	Exit "$Msg"
    elif [ $ExitCode -eq $OK ]; then
	UserList "$Option"
    elif [ $ExitCode -eq $ESC ]; then
	echo Esc pressed. >&2
    fi

}


UserList(){
    Option=$1
    if [ $Option -eq 1 ]; then
	Users=$(grep -v nologin /etc/passwd | sed '1,2d' | awk -F\: '{ print $1 " " $3 " off"}')
	SelectedUser=$(dialog --title "POST ANNOUNCEMENT" --extra-button --extra-label "ALL" --single-quoted --clear \
	    --checklist "Please choose who you want to post" 20 50 5 \
	    $Users \
	    2>&1 > /dev/tty)

        ExitCode=$?
	if [ $ExitCode -eq $CANCEL ]; then
	    EntrancePage
        elif [ $ExitCode -eq $ALL ]; then
	    AllUsers=$(echo "$Users" | awk '{ print $1 }' | tr '\n' ',')
	    AnnounceMsg "$AllUsers"
    	elif [ $ExitCode -eq $OK ]; then
	    AnnounceMsg "$SelectedUser"
        elif [ $ExitCode -eq $ESC ]; then
	    echo Esc pressed. >&2
	fi
    else
	who | awk '{ print $1 }' | uniq > tmp.txt
	grep -v nologin /etc/passwd | sed '1,2d' | awk -F\: '{ print $1 " " $3 }' > tmp2.txt
	Users=$(awk 'NR==FNR{a[$1]="[*]";next}{print $0""a[$1]}' tmp.txt tmp2.txt)
	rm tmp.txt tmp2.txt

	SelectedUser=$(dialog --ok-label "SELECT" --cancel-label "EXIT" --clear \
	    --menu "User Info Panel" 20 50 10 \
	    $Users \
	    2>&1 > /dev/tty)
	ExitCode=$?
	if [ $ExitCode -eq $CANCEL ]; then
	    EntrancePage
	elif [ $ExitCode -eq $ESC ]; then
	    echo Esc pressed. >&2
	    exit 1
	fi

	Ret=$(echo "$SudoPassword" | sudo -S grep LOCKED /etc/master.passwd | grep "$SelectedUser")
	ExitCode=$?
	if [ $ExitCode -eq $CANCEL ]; then
	    LockUnlock=$(echo "UnLocked")
	else
	    LockUnlock=$(echo "Locked")
	fi

	UserAction "$SelectedUser" "$LockUnlock"
    fi

}

AnnounceMsg(){
    SelectedUser=$1
    Msg=$(dialog --title "Post an announcement" --clear \
        --inputbox "Enter your messages:" 20 50 \
	2>&1 > /dev/tty)
    
    ExitCode=$?
    if [ $ExitCode -eq $OK ]; then
	UserList=$(echo "$SelectedUser" | tr ' ' ',')
	echo "$SudoPassword" | sudo -S pw groupadd SA_34 -M "$UserList"
	echo "$Msg" | wall -g SA_34
	echo "$SudoPassword" | sudo -S pw groupdel SA_34
    elif [ $ExitCode -eq $ESC ]; then
	echo Esc pressed. >&2
    fi
    EntrancePage
}

UserAction(){
    SelectedUser=$1
    LockUnlock=$2

    if [ $LockUnlock = "UnLocked" ]; then
	Lockmsg=$(echo "LOCK")
    else
	Lockmsg=$(echo "UNLOCK")
    fi
        
    Action=$(dialog --cancel-label "EXIT" --clear \
        --menu "User vagrant" 20 50 10 \
	1 "$Lockmsg IT" \
	2 "GROUP INFO" \
	3 "PORT INFO" \
	4 "LOGIN INFO" \
	5 "SUDO LOG" \
	2>&1 > /dev/tty)

    ExitCode=$?
    if [ $ExitCode -eq $OK ]; then
        case $Action in
	    1)
	    LockUnLockCheck "$SelectedUser" "$LockUnlock"
	    ;;
	    2)
	    GroupInfo "$SelectedUser" "$LockUnlock"
	    ;;
	    3)
	    PortInfo "$SelectedUser" "$LockUnlock"
	    ;;
	    4)
	    LoginHistory "$SelectedUser" "$LockUnlock"
	    ;;
            5)
	    SudoLog "$SelectedUser" "$LockUnlock"
	    ;;
        esac
    elif [ $ExitCode -eq $CANCEL ]; then
	UserList 2
    elif [ $ExitCode -eq $ESC ]; then
	echo Esc pressed. >&2
    fi
}

LockUnLockCheck(){
    SelectedUser=$1
    LockUnlock=$2
    
    if [ $LockUnlock = "UnLocked" ]; then
	Lockmsg=$(echo "LOCK")
    else
	Lockmsg=$(echo "UNLOCK")
    fi

    dialog --title "$Lockmsg IT" --clear \
    --yesno "Are you sure you want to do this?" 20 40

    ExitCode=$?
    if [ $ExitCode -eq $OK ]; then
	LockUnlockMsg $SelectedUser $LockUnlock
    elif [ $ExitCode -eq $CANCEL ]; then 
	UserAction $SelectedUser $LockUnlock
    elif [ $ExitCode -eq $ESC ]; then
	echo Esc pressed. >&2
    fi
}

LockUnlockMsg(){
    SelectedUser=$1
    LockUnlock=$2

    if [ $LockUnlock = "UnLocked" ]; then
	Lockmsg=$(echo "LOCK")
	LockUnlock=$(echo "Locked")
	echo "$SudoPassword" | sudo -S pw lock "$SelectedUser"
    else
	Lockmsg=$(echo "UNLOCK")
	LockUnlock=$(echo "UnLocked")
	echo "$SudoPassword" | sudo -S pw unlock "$SelectedUser"
    fi

    dialog --title "$Lockmsg IT" --clear \
    --msgbox "$Lockmsg SUCCEED!" 20 40

    ExitCode=$?
    if [ $ExitCode -eq $ESC ]; then
        echo Esc pressed. >&2
    fi
    
    UserAction $SelectedUser $LockUnlock
}

GroupInfo(){
    SelectedUser=$1
    LockUnlock=$2
    data=$(groups "$SelectedUser" | tr ' ' '\n' | xargs -I % -n1 grep % /etc/group | sort -t: -k3 | awk -F\: 'BEGIN {print "GROUP_ID GROUP_NAME"} {print $3 " " $1}')
    
    dialog --title "GROUP" --yes-label "OK" --no-label "EXPORT" --clear \
    --yesno "$data" 20 40

    ExitCode=$?
    if [ $ExitCode -eq $OK ]; then
	UserAction "$SelectedUser" "$LockUnlock"
    elif [ $ExitCode -eq $CANCEL ]; then
	Export "$SelectedUser" "$LockUnlock" "$data"
	GroupInfo "$SelectedUser" "$LockUnlock"
    elif [ $ExitCode -eq $ESC ]; then
	echo Esc pressed. >&2
    fi        
}

PortInfo(){
    SelectedUser=$1
    LockUnlock=$2
    data=$(sockstat -4 | sed 1d | sed '/'"$SelectedUser"'/!d' | awk '{ print $3 " " $5 "_" $6 }')
    
    check=$(sockstat -4 | sed 1d | sed '/'"$SelectedUser"'/!d' | grep "$SelectedUser")
    ExitCode=$?
    if [ $ExitCode -eq $CANCEL ]; then
	NoPortInfo "$SelectedUser" "$LockUnlock"
    fi

    Port=$(dialog --title "Port INFO(PID and Port)" --clear \
        --menu " " 20 50 10 \
	$data \
	2>&1 > /dev/tty)

    ExitCode=$?
    if [ $ExitCode -eq $OK ]; then
	ProcessState "$SelectedUser" "$Port" "$LockUnlock"
    elif [ $ExitCode -eq $ESC ]; then
	echo Esc pressed. >&2
    elif [ $ExitCode -eq $CANCEL ]; then
        UserAction "$SelectedUser" "$LockUnlock"
    fi
}

NoPortInfo(){
    SelectedUser=$1
    LockUnlock=$2
    dialog --title "No Port Info" --clear \
    --msgbox "There is no port info for this user.\nPlease select other users." 20 40
	
    ExitCode=$?
    if [ $ExitCode -eq $ESC ]; then
	echo Esc pressed. >&2
    fi

    UserAction "$SelectedUser" "$LockUnlock"
}

ProcessState(){
    SelectedUser=$1
    port=$2
    LockUnlock=$3
    data=$(ps -o user -o pid -o ppid -o state -o %cpu -o %mem -o command "$port" | sed 1d | awk '
    	    BEGIN { 
	    	split("USER PID PPID STAT %CPU %MEM COMMAND", keyword, " ") 
	    } 
            {
		command = $7
		for (i=8; i<= NF; i++) {
		    command = command " " $i
		}
		for(i=1; i<=6; i++) { 
		    print keyword[i] " " $i 
		}
	        print keyword[7] " " command 
	    }')
    dialog --title "PROCESS STATE: $port" --yes-label "OK" --no-label "EXPORT" --clear \
    --yesno "$data" 30 50

    ExitCode=$?
    if [ $ExitCode -eq $OK ]; then
	PortInfo "$SelectedUser" "$LockUnlock"
    elif [ $ExitCode -eq $CANCEL ]; then
	Export "$SelectedUser" "$LockUnlock" "$data"
	ProcessState "$SelectedUser" "$Port" "$LockUnlock"
    elif [ $ExitCode -eq $ESC ]; then
	echo Esc pressed. >&2
    fi  
}

LoginHistory(){
    SelectedUser=$1
    LockUnlock=$2
    # Select the recent 10 login history of specified user
    History=$( last | sed '/^.*[\.].*[\.].*$/!d' | sed '/'"$SelectedUser"'/!d' | sed '11,$d' | awk 'BEGIN {print "DATE IP"}{ print $4 " " $5 " " $6 " " $7 " " $3 }')

    dialog --title "LOGIN HISTORY" --yes-label "OK" --no-label "EXPORT" --clear \
    --yesno "$History" 50 100

    ExitCode=$?
    if [ $ExitCode -eq $OK ]; then
	UserAction "$SelectedUser" "$LockUnlock"
    elif [ $ExitCode -eq $CANCEL ]; then
	Export "$SelectedUser" "$LockUnlock" "$History"
	LoginHistory "$SelectedUser" "$LockUnlock"
    elif [ $ExitCode -eq $ESC ]; then
	echo Esc pressed. >&2
    fi
}

SudoLog(){
    SelectedUser=$1
    LockUnlock=$2

    # Select the log in recent 30 days and select sudo log
    msg=$(echo "$SudoPassword" | sudo -S cat /var/log/auth.log | awk '{
    	logdate = $1 " " $2 " " $3
	srand()
	"date -j -f \"%b %d %T\" \"" logdate "\" +%s" | getline epoch
	if((srand() - epoch) <= 2592000) {
	    print $0
	}
    }' | sed -e 's/^\(...[^a-zA-Z]*\).*sudo.*:[ ]*\([^ ]*\) :.*COMMAND=\(.*\)/\2 used sudo to do `\3` on \1/' -e '/used sudo/!d' -e '/'^"$SelectedUser"'/!d')

    dialog --title "SUDO LOG" --yes-label "OK" --no-label "EXPORT" --clear \
    --yesno "$msg" 60 150
    
    ExitCode=$?
    if [ $ExitCode -eq $OK ]; then
        UserAction "$SelectedUser" "$LockUnlock"
    elif [ $ExitCode -eq $CANCEL ]; then
	Export "$SelectedUser" "$LockUnlock" "$msg"
	SudoLog "$SelectedUser" "$LockUnlock"
    elif [ $ExitCode -eq $ESC ]; then
	echo Esc pressed. >&2
    fi
}

Export(){
    SelectedUser=$1
    LockUnlock=$2
    data=$3
    CurrentUser=$(who am I | awk '{print $1}')

    Path=$(dialog --title "Export to file" --clear \
	--inputbox "Enter the path:" 10 60 \
	2>&1 > /dev/tty)
    ExitCode=$?

    if [ $ExitCode -eq $OK ]; then
	homedir=$(grep "$SelectedUser" /etc/passwd | awk -F\: '{print $6}')
	tmp=$(echo "$Path" | grep "$homedir")
	ExitCode=$?
	if [ $ExitCode -eq $OK ]; then
	    echo "$data" > $Path
	else
	    dir=${homedir}"/"${Path}
	    echo "$data" > $dir
	fi
    elif [ $ExitCode -eq $ESC ]; then
	echo Esc pressed. >&2
    fi
    return 0
}

Exit(){
    clear
    echo "$1"
    exit
}

CtrlC=$(echo "Ctrl + C pressed.")
trap 'Exit "$CtrlC"' SIGINT
SudoPassword=$(dialog --title "Enter Password" --clear \
	--passwordbox "In order to use some sudo commands, please enter your password first: " 10 80 \
	2>&1 > /dev/tty)
ExitCode=$?
if [ $ExitCode -eq $CANCEL ]; then
    Exit "EXIT."
elif [ $ExitCode -eq $OK ]; then
    EntrancePage
else
    echo Esc pressed. >&2
fi
