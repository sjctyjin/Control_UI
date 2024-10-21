        //全域變數
        var canvas, context, image;
        var arr = new Array(); 
        var clone_panel;
        var availableTags = [];

        function delay(n){
            return new Promise(function(resolve){
                setTimeout(resolve,n*1000);
            });
        }

        //載入網頁時,同時載入圖片
        async function myAsyncFunction(seleceidx){
                    
            $("#imgarea").css("background-image","url("+document.getElementById("filelist").options[seleceidx].text+")");
            
            $("#label_img").attr("src",document.getElementById("filelist").options[seleceidx].text  );
            await delay(0.1);//delay 0.1 sec
            
            console.log(Math.round(parseFloat($('#label_img').css("width"))));
            document.getElementById("pCanvas").width = Math.round(parseFloat($('#label_img').css("width")));
            document.getElementById("pCanvas").height = Math.round(parseFloat($('#label_img').css("height")));
            check_cav = 1;
            console.log("Loding........ 完成");
            $(".store_label").css("height", Math.round(parseFloat($('#label_img').css("height")))+"px");
        }

        async function loge(){
            if(document.getElementById('class_by_keyword') != document.activeElement){

                switch (event.key) {
                    case "a":
                        if(document.getElementById("filelist").options.length > 0){
                        
                            var idx = document.getElementById("filelist").selectedIndex;
                            if(parseInt(idx) > 0 ){

                                //Step1.先儲存當前label結果，後刪除lable
                                if(document.querySelectorAll(".label_board").length != 0){
                                    
                                    ajaxStore();
                                    $('.label_board').remove()
                                }
                                //Step2 . 載入上一張

                                document.getElementById("filelist").selectedIndex = parseInt(idx-1);
                                myAsyncFunction(parseInt(idx-1));
                                ajaxRead();//若有結果,就秀出上次label結果

                                // $("#imgarea").css("background-image","url("+document.getElementById("filelist").options[parseInt(idx-1)].text+")")
                                // $("#label_img").attr("src",document.getElementById("filelist").options[parseInt(idx-1)].text  );
                            }
                            await delay(0.5);//delay 0.1 sec
                            draw_all_rect();
                    
                        }
                        break;
                    case "d":
                        if(document.getElementById("filelist").options.length > 0){
                            var labelarea = document.getElementsByClassName("store_label")[0];
                            var labeldiv = 0;
                            var idx = document.getElementById("filelist").selectedIndex;
                        
                            if(parseInt(idx) < document.getElementById("filelist").options.length ){

                                
                                        
                                //Step1.先儲存當前label結果，後刪除lable
                                if(document.querySelectorAll(".label_board").length != 0){
                                    
                                    ajaxStore();
                                    $('.label_board').remove()
                                }
                                //Step2 . 載入下一張
                                document.getElementById("filelist").selectedIndex = parseInt(idx+1);
                                myAsyncFunction(parseInt(idx+1));   

                                ajaxRead();
                            }
                            await delay(0.5);//delay 0.1 sec
                            draw_all_rect();
                        }
                        break;
                    case "ArrowLeft":
                        console.log("ArrowLeft");
                        break;
                    case "ArrowRight":
                        console.log("ArrowRight");
                        break;
                    default:
                        console.log(event.key, event.keyCode);
                        key_keep = event.key;
                        console.log("顯示"+key_keep);
                
                    //document.getElementById("textId").value = key_keep;
                    //return; 

                }
            }
        }

        

       
        function draw_all_rect(){
            arr = new Array();
            var labelarea = document.getElementsByClassName("store_label")[0];
            var divc = 0;
            
            for(var divn = 0;divn < labelarea.childNodes.length;divn++){
                if(labelarea.childNodes[divn].nodeName == "DIV"){
                    
                    if(divc != 0){
                        arr.push(divn);
                    }
                    divc++;
                }          
            } 
            console.log("測試項1 : ",arr)
            //依陣列值畫圖框
            for(var m = 0;m < arr.length;m++){
                console.log("測試項2 : ",arr[m])
                
                var thiss = labelarea.childNodes[arr[m]];
                        var bxmin = thiss.children[0].children[1].getElementsByTagName('b')[0].innerHTML;
                        var bymin = thiss.children[0].children[1].getElementsByTagName('b')[1].innerHTML;
                        var bmax = thiss.children[0].children[1].getElementsByTagName('b')[2].innerHTML;
                        var bmin = thiss.children[0].children[1].getElementsByTagName('b')[3].innerHTML;
                        var bh = thiss.children[0].children[1].getElementsByTagName('b')[4].innerHTML;
                        var bw = thiss.children[0].children[1].getElementsByTagName('b')[5].innerHTML;
                        //context.clearRect(0,0,canvas.width,canvas.height);
                        //context.strokeStyle = "rgba(255,0,0,0.3)";
                        context.fillStyle = "rgba(0,0,255,0.2)";
                        context.beginPath();
                        context.rect(bxmin,bymin,bw,bh);
                        context.strokeRect(bxmin,bymin,bw,bh);
                        context.fill();
                        //console.log(divc);
            }

            return arr[m];


        }
        
        // function dvd(q){
        //         var bxmin = this.children[0].children[1].getElementsByTagName('b')[0].innerHTML;
        //         var bymin = this.children[0].children[1].getElementsByTagName('b')[1].innerHTML;
        //         var bmax = this.children[0].children[1].getElementsByTagName('b')[2].innerHTML;
        //         var bmin = this.children[0].children[1].getElementsByTagName('b')[3].innerHTML;
        //         var bh = this.children[0].children[1].getElementsByTagName('b')[4].innerHTML;
        //         var bw = this.children[0].children[1].getElementsByTagName('b')[5].innerHTML;
        //         context.clearRect(0,0,canvas.width,canvas.height);
        //         context.strokeRect(bxmin,bymin,bw,bh);
        //         // console.log(bxmin);
        //     };

        function check_pass(val){
            //console.log(val.nodeName);
            if(val.key == "Enter" || val.nodeName == "BUTTON"){
                
                d = document.getElementsByClassName('label_card')[0].cloneNode(true);
                //顯示字卡
                d.style.display = "block";
                classname = document.getElementById('label_class').querySelector('input').value;

                //標註名稱-次序
                d.children[0].children[0].children[0].children[0].innerHTML = classname;
                if(stopx < startx){
                    stop_X = startx;
                    start_X = stopx;
                }else{
                    stop_X = stopx;
                    start_X = startx;
                }

                if(stopy < starty){
                    stop_Y = starty;
                    start_Y = stopy;
                }else{
                    stop_Y = stopy;
                    start_Y = starty;
                }
                d.children[0].children[1].getElementsByTagName('b')[0].innerHTML= start_X;
                d.children[0].children[1].getElementsByTagName('b')[1].innerHTML= start_Y;
                d.children[0].children[1].getElementsByTagName('b')[2].innerHTML= stop_X;
                d.children[0].children[1].getElementsByTagName('b')[3].innerHTML= stop_Y;
                d.children[0].children[1].getElementsByTagName('b')[4].innerHTML= Math.abs(Math.round(parseFloat(h)));
                d.children[0].children[1].getElementsByTagName('b')[5].innerHTML= Math.abs(Math.round(parseFloat(w)));
                d.setAttribute("class","col-md-8 label_card label_board" );
                d.onmouseenter = function(){
                    var bxmin = this.children[0].children[1].getElementsByTagName('b')[0].innerHTML;
                    var bymin = this.children[0].children[1].getElementsByTagName('b')[1].innerHTML;
                    var bmax = this.children[0].children[1].getElementsByTagName('b')[2].innerHTML;
                    var bmin = this.children[0].children[1].getElementsByTagName('b')[3].innerHTML;
                    var bh = this.children[0].children[1].getElementsByTagName('b')[4].innerHTML;
                    var bw = this.children[0].children[1].getElementsByTagName('b')[5].innerHTML;
                    context.clearRect(0,0,canvas.width,canvas.height);
                    context.beginPath();
                    //清空繪圖區域
                    //context.clear(context.COLOR_BUFFER_BIT);
                    //context.clearColor(0.0, 0.5, 0.5, 1.0);
                    context.strokeStyle = "rgba(0,255,0,0.3)";
                    context.fillStyle = "rgba(0,0,255,0.3)";
                    
                    context.rect(bxmin,bymin,bw,bh);
                    context.strokeRect(bxmin,bymin,bw,bh);
                    context.fill();

                    

                    // console.log(bxmin);
                };
                d.onmouseleave = function(){
                    // context.clear(context.COLOR_BUFFER_BIT);
                    // context.clearColor(0.0, 0.5, 0.5, 1.0);
                    
                    draw_all_rect();
                };
            
                document.getElementsByClassName("store_label")[0].appendChild(d);
                //標註panel
                document.getElementById("label_class").style.display = "none";   
                //清除畫痕
                context.clearRect(0,0,canvas.width,canvas.height);   
                console.log(d);
                draw_all_rect();

                //檢查select是否存在該值-----------------------------------------------------

                var select = document.getElementById("class_list");
                    var inputValue = document.getElementById("class_by_keyword").value;
                    var options = select.options;
                    var found = false;
                    console.log(found);
                    for (var i = 0; i < options.length; i++) {
                        if (options[i].value === inputValue) {
                        found = true;
                        break;
                        }
                    }

                    if (!found) {
                        var option = document.createElement("option");
                        option.value = inputValue;
                        option.text = inputValue;
                        select.add(option);
                        console.log('加入');
                        availableTags.push(inputValue);
                            var csrft = $("[name='csrfmiddlewaretoken']").val();
                        var filepath = document.getElementById("filelist").options[document.getElementById("filelist").selectedIndex].text

                        $.ajax({
                            url:"classname",
                            type:"POST",
                            contentType: "application/x-www-form-urlencoded;charset=UTF-8",
                            //Django用法 Post接收contentType 使用x-www-form-urlencoded
                            //一般使用 application/json;
                            data:                   
                            {   
                                "tags":inputValue,   
                                "filepath":filepath,
                                "csrfmiddlewaretoken":csrft,             
                            }
                            ,dataType: 'json',

                                success: function(data){
                                // console.log('ajax result:')
                                console.log(data)
                                //alert('新增成功');
                            }
                        })

                    }

                    document.getElementsByClassName("form-select")[0].querySelectorAll('option').forEach(option => {
                        if($("#class_by_keyword").val() == option.textContent){
                            document.getElementById("class_list").selectedIndex = option.index;
                        }
                    });
            }
            // else
            // {
            //     check_no();
            // }
            //console.log(val.parentElement.parentElement.parentElement.querySelector('input').value);
            
        }

        function check_no(val){
            document.getElementById("label_class").style.display = "none";   
            //清除畫痕
            context.clearRect(0,0,canvas.width,canvas.height);   
        }
        function getlist(val){
            document.getElementById('label_class').querySelector('input').value = val.options[val.selectedIndex].text;
            // console.log(val.options[val.selectedIndex].text)
            
        }
        //file list
        async function getlist2(val){
            //document.getElementById('label_class').querySelector('input').value = val.options[val.selectedIndex].text;
            console.log(val.options[val.selectedIndex].text);
            
            if(document.querySelectorAll(".label_board").length != 0){
                //儲存當前label結果
                //ajaxStore();
                $('.label_board').remove()

            }
            myAsyncFunction(val.selectedIndex);
            $(".store_label").css("height", Math.round(parseFloat($('#label_img').css("height")))+"px");
            ajaxRead();
            await delay(0.5);//delay 0.1 sec
            draw_all_rect();

        }

        function funs(){
            var d = document.getElementById('updfiles').files.length;
            document.getElementById('filecount').innerHTML = "已選擇"+d+"個檔案";
            document.getElementById('dir_name').style.display="block";
            console.log(d);
        }

        


        function ajaxStore(){
            var csrft = $("[name='csrfmiddlewaretoken']").val();
            var filepath = document.getElementById("filelist").options[document.getElementById("filelist").selectedIndex].text
            var imgwidth = $('#label_img').width();
            var imgheight = $('#label_img').height();
            var labelarea = document.getElementsByClassName("store_label")[0];
            var abrr = new Array();
            optionTexts = [];
            document.getElementsByClassName("form-select")[0].querySelectorAll('option').forEach(option => {
                optionTexts.push(option.textContent);
            });
            for(var m = 0;m < arr.length;m++){
                console.log("將結果儲存 : ",arr[m])
                
                        var thiss = labelarea.childNodes[arr[m]];
                        var labelname = thiss.children[0].children[0].getElementsByTagName('p')[0].innerHTML;
                        var bxmin = thiss.children[0].children[1].getElementsByTagName('b')[0].innerHTML;
                        var bymin = thiss.children[0].children[1].getElementsByTagName('b')[1].innerHTML;
                        var bmax = thiss.children[0].children[1].getElementsByTagName('b')[2].innerHTML;
                        var bmin = thiss.children[0].children[1].getElementsByTagName('b')[3].innerHTML;
                        var bh = thiss.children[0].children[1].getElementsByTagName('b')[4].innerHTML;
                        var bw = thiss.children[0].children[1].getElementsByTagName('b')[5].innerHTML;
                        abrr.push(labelname,bxmin,bymin,bmax,bmin,bh,bw);//標註結果資料
            }
            $.ajax({
                url:"list",
                type:"POST",
                contentType: "application/x-www-form-urlencoded;charset=UTF-8",
                //Django用法 Post接收contentType 使用x-www-form-urlencoded
                //一般使用 application/json;
                data:                   
                {   
                    "filepath":filepath,
                    "imgwidth":imgwidth,
                    "imgheight":imgheight,
                    label:abrr,
                    labellist:optionTexts,
                    "csrfmiddlewaretoken":csrft,
                                    
                }
                ,dataType: 'json',

                    success: function(data){
                    // console.log('ajax result:')
                    console.log(data)
                    //alert('新增成功');
                }
            })

        }
        function ajaxRead(){//讀取已存在的標註框
            var csrft = $("[name='csrfmiddlewaretoken']").val();
            var filepath = document.getElementById("filelist").options[document.getElementById("filelist").selectedIndex].text

            $.ajax({
                url:"read",
                type:"POST",
                contentType: "application/x-www-form-urlencoded;charset=UTF-8",
                //Django用法 Post接收contentType 使用x-www-form-urlencoded
                //一般使用 application/json;
                data: 
                {   
                    "filepath":filepath,
                    "csrfmiddlewaretoken":csrft,
                }
                ,dataType: 'json',

                    success: function(data){
                    // console.log('ajax result:')
                        data['data'].forEach(option => {
                            console.log(option[0]);

                            d = document.getElementsByClassName('label_card')[0].cloneNode(true);
                            //顯示字卡
                            d.style.display = "block";
                            //標註名稱-次序
                            d.children[0].children[0].children[0].children[0].innerHTML = option[0];
                            d.children[0].children[1].getElementsByTagName('b')[0].innerHTML= option[1];
                            d.children[0].children[1].getElementsByTagName('b')[1].innerHTML= option[2];
                            d.children[0].children[1].getElementsByTagName('b')[2].innerHTML= option[3];
                            d.children[0].children[1].getElementsByTagName('b')[3].innerHTML= option[4];
                            d.children[0].children[1].getElementsByTagName('b')[4].innerHTML= option[5];
                            d.children[0].children[1].getElementsByTagName('b')[5].innerHTML= option[6];
                            d.setAttribute("class","col-md-8 label_card label_board" );
                            d.onmouseenter = function(){
                                var bxmin = this.children[0].children[1].getElementsByTagName('b')[0].innerHTML;
                                var bymin = this.children[0].children[1].getElementsByTagName('b')[1].innerHTML;
                                var bmax = this.children[0].children[1].getElementsByTagName('b')[2].innerHTML;
                                var bmin = this.children[0].children[1].getElementsByTagName('b')[3].innerHTML;
                                var bh = this.children[0].children[1].getElementsByTagName('b')[4].innerHTML;
                                var bw = this.children[0].children[1].getElementsByTagName('b')[5].innerHTML;
                                context.clearRect(0,0,canvas.width,canvas.height);
                                context.beginPath();
                                //清空繪圖區域
                                //context.clear(context.COLOR_BUFFER_BIT);
                                //context.clearColor(0.0, 0.5, 0.5, 1.0);
                                context.strokeStyle = "rgba(0,255,0,0.3)";
                                context.fillStyle = "rgba(0,0,255,0.3)";
                                context.rect(bxmin,bymin,bw,bh);
                                context.strokeRect(bxmin,bymin,bw,bh);
                                context.fill();
                                // console.log(bxmin);
                        };
                        d.onmouseleave = function(){
                            // context.clear(context.COLOR_BUFFER_BIT);
                            // context.clearColor(0.0, 0.5, 0.5, 1.0);
                            
                            draw_all_rect();
                        };
                    
                        document.getElementsByClassName("store_label")[0].appendChild(d);
                        //標註panel
                        document.getElementById("label_class").style.display = "none";   
                        //清除畫痕
                        //context.clearRect(0,0,canvas.width,canvas.height);   
                        //console.log(d);
                        

                        });
                        console.log(data['data']);
                    }
            })

        }


        function check(){
            if(document.getElementById('data_set_name').value == ""){
                alert("請填入資料夾名稱");

            }
            else{
                var userForm = document.getElementById("userForm");
                userForm.submit();
                return true;
            }
        }
