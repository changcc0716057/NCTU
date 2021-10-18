#/bin/bash

OK=0
CANCEL=1
ESC=255
BACKTITLE="Cody Blog - Example"

Menu(){
    while :
    do {
        Selection=$(dialog --title "CodyBlog - Menu" --clear \
            --backtitle "$BACKTITLE" \
            --cancel-label "Exit" \
            --menu "Choose one" 12 45 5 \
            1 "Test 01" \
            2 "Test 02" \
            2>&1 > /dev/tty)

        result=$?
        if [ $result -eq $OK ]; then
            Select $Selection
        elif [ $result -eq $CANCEL ] || [ $result -eq $ESC ]; then
            Exit
        fi
    } done
}

Select(){
    Choice=$1

    case $Choice in
        1)  # Option 1
            Option1
            ;;
        2)  # Option 2
            Option2
            ;;
    esac
}

Option1(){
    while :
    do {
        exec 3>&1
        Input=$(dialog --title "$Title" --clear \
            --backtitle "$BACKTITLE" \
            --form "Please input:" 10 70 5  \
            "Input1:" 1  2 "" 1  14  48  0 \
            "Input2:" 2  2 "" 2  14  48  0 \
            2>&1 1>&3)
        result=$?
        exec 3>&-

        IFS=$'\n'
        Values=($Input)
        unset IFS

        if [ $result -eq $CANCEL ] || [ $result -eq $ESC ]; then
            break
        elif [ $result -eq $OK ]; then
            MsgBox "${Values[0]}" "${Values[1]}"
            if [ $result -eq $OK ]; then
                break
            fi
        fi
    } done
}

Option2(){
    while :
    do {
        Number=$(dialog --title 'Option 2' --clear \
        --backtitle "$BACKTITLE" \
        --radiolist 'Select Item:' 15 70 2 \
        1 Radiolist1 off \
        2 Radiolist2 on \
        2>&1 > /dev/tty)
        result=$?
        
        if [ $result -eq $CANCEL ] || [ $result -eq $ESC ]; then
            break
        elif [ $result -eq $OK ]; then
            MsgBox "$Number"
            break
        fi
    } done
}

MsgBox(){
    Input1=$1
    Input2=$2

    Msg="$Input1\n"
    Msg="${Msg}$Input2"

    dialog --title "MsgBox" --clear \
    --backtitle "$BACKTITLE" \
    --yesno "$Msg" 10 70

    result=$?
}

Exit(){
    clear
    echo "Program Terminated."
    exit
}

Menu
