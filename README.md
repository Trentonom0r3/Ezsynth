TODO: 
- Further Refactor to include index with original input, so user can pass numpy arrays directly.

- Further optimize performance.
  - I know parallel processing w/ebsynth is possible, its simply a matter of figuring out how exactly to set things up.
  - External assistance regarding this would certainly be nice. 
  - cProfile output:

  ```
  Sat Oct 14 04:48:50 2023    cumtime.profile

         616839 function calls (545961 primitive calls) in 272.149 seconds

   Ordered by: cumulative time
   List reduced from 911 to 50 due to restriction <50>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      5/1    0.001    0.000  272.148  272.148 {built-in method builtins.exec}
        1    0.189    0.189  272.147  272.147 <string>:1(<module>)
        1    0.003    0.003  271.958  271.958 c:\Users\tjerf\Desktop\Refactor_test\testme.py:8(main)
        1    0.079    0.079  245.524  245.524 c:\Users\tjerf\Desktop\Refactor_test\utils\utils.py:144(run)
        1    0.011    0.011  245.445  245.445 c:\Users\tjerf\Desktop\Refactor_test\utils\utils.py:147(process)
      330  243.737    0.739  243.737    0.739 {method 'acquire' of '_thread.lock' objects}
        3    0.000    0.000  217.729   72.576 C:\Python311\Lib\concurrent\futures\_base.py:646(__exit__)
        4    0.000    0.000  217.728   54.432 C:\Python311\Lib\threading.py:1080(join)
        4    0.001    0.000  217.728   54.432 C:\Python311\Lib\threading.py:1118(_wait_for_tstate_lock)
        2    0.000    0.000  128.005   64.002 C:\Python311\Lib\concurrent\futures\thread.py:216(shutdown)
        1    0.007    0.007  117.426  117.426 c:\Users\tjerf\Desktop\Refactor_test\utils\blend\blender.py:98(__call__)
        1    0.000    0.000   90.070   90.070 c:\Users\tjerf\Desktop\Refactor_test\utils\blend\blender.py:93(_reconstruct)
        1    0.000    0.000   90.070   90.070 c:\Users\tjerf\Desktop\Refactor_test\utils\blend\reconstruction.py:21(__call__)
        1    0.274    0.274   90.070   90.070 c:\Users\tjerf\Desktop\Refactor_test\utils\blend\reconstruction.py:24(_create)
        1    0.000    0.000   89.724   89.724 C:\Python311\Lib\concurrent\futures\process.py:821(shutdown)
        1    0.001    0.001   26.017   26.017 c:\Users\tjerf\Desktop\Refactor_test\utils\blend\blender.py:84(_hist_blend)
      106    0.001    0.000   26.012    0.245 C:\Python311\Lib\threading.py:288(wait)
      200    0.002    0.000   26.007    0.130 C:\Python311\Lib\concurrent\futures\_base.py:428(result)
        1    0.000    0.000   25.097   25.097 c:\Users\tjerf\Desktop\Refactor_test\utils\utils.py:118(__init__)
        1    0.001    0.001   24.383   24.383 c:\Users\tjerf\Desktop\Refactor_test\utils\guides\guides.py:67(create_all_guides)
        1    0.001    0.001   16.373   16.373 c:\Users\tjerf\Desktop\Refactor_test\utils\guides\guides.py:74(<listcomp>)
      100    0.003    0.000   16.372    0.164 c:\Users\tjerf\Desktop\Refactor_test\utils\flow_utils\OpticalFlow.py:103(__iter__)
       99    0.034    0.000   16.369    0.165 c:\Users\tjerf\Desktop\Refactor_test\utils\flow_utils\OpticalFlow.py:91(_compute_flow)
   34155/99    0.042    0.000    9.461    0.096 C:\Users\tjerf\AppData\Roaming\Python\Python311\site-packages\torch\nn\modules\module.py:1498(_wrapped_call_impl)
   34155/99    0.105    0.000    9.461    0.096 C:\Users\tjerf\AppData\Roaming\Python\Python311\site-packages\torch\nn\modules\module.py:1504(_call_impl)
         99    0.150    0.002    9.454    0.095 c:\Users\tjerf\Desktop\Refactor_test\utils\flow_utils\core\raft.py:86(forward)
        199    6.267    0.031    6.267    0.031 {method 'cpu' of 'torch._C._TensorBase' objects}
        990    0.076    0.000    4.844    0.005 c:\Users\tjerf\Desktop\Refactor_test\utils\flow_utils\core\update.py:127(forward)
          1    0.000    0.000    4.144    4.144 c:\Users\tjerf\Desktop\Refactor_test\utils\guides\guides.py:122(__call__)
          1    0.000    0.000    4.144    4.144 c:\Users\tjerf\Desktop\Refactor_test\utils\guides\guides.py:133(_compute_edge)
          1    0.000    0.000    4.144    4.144 c:\Users\tjerf\Desktop\Refactor_test\utils\guides\guides.py:137(<listcomp>)
        100    0.000    0.000    4.144    0.041 c:\Users\tjerf\Desktop\Refactor_test\utils\guides\guides.py:139(_create)
        100    0.263    0.003    4.143    0.041 c:\Users\tjerf\Desktop\Refactor_test\utils\guides\edge_detection.py:77(compute_edge)
        294    0.633    0.002    3.711    0.013 c:\Users\tjerf\Desktop\Refactor_test\utils\flow_utils\warp.py:31(run_warping)
          2    0.000    0.000    3.638    1.819 c:\Users\tjerf\Desktop\Refactor_test\utils\guides\guides.py:170(__call__)
          2    0.000    0.000    3.638    1.819 c:\Users\tjerf\Desktop\Refactor_test\utils\guides\guides.py:204(_create)
          2    0.255    0.127    3.638    1.819 c:\Users\tjerf\Desktop\Refactor_test\utils\guides\guides.py:192(_create_g_pos_from_flow)
      18018    0.054    0.000    3.559    0.000 C:\Users\tjerf\AppData\Roaming\Python\Python311\site-packages\torch\nn\modules\conv.py:462(forward) 
      18018    0.025    0.000    3.486    0.000 C:\Users\tjerf\AppData\Roaming\Python\Python311\site-packages\torch\nn\modules\conv.py:454(_conv_forward)
      18018    3.461    0.000    3.461    0.000 {built-in method torch.conv2d}
        100    0.002    0.000    3.291    0.033 C:\Users\tjerf\AppData\Roaming\Python\Python311\site-packages\phycv\page_gpu.py:182(run)
        990    0.436    0.000    3.113    0.003 c:\Users\tjerf\Desktop\Refactor_test\utils\flow_utils\core\corr.py:29(__call__)
        198    0.256    0.001    3.040    0.015 c:\Users\tjerf\Desktop\Refactor_test\utils\guides\guides.py:173(_create_and_warp_coord_map)
        990    0.487    0.000    2.478    0.003 c:\Users\tjerf\Desktop\Refactor_test\utils\flow_utils\core\update.py:45(forward)
        294    0.493    0.002    2.064    0.007 c:\Users\tjerf\Desktop\Refactor_test\utils\flow_utils\warp.py:23(_warp)
        100    0.008    0.000    1.843    0.018 C:\Users\tjerf\AppData\Roaming\Python\Python311\site-packages\phycv\page_gpu.py:125(apply_kernel)   
       3960    1.245    0.000    1.788    0.000 c:\Users\tjerf\Desktop\Refactor_test\utils\flow_utils\core\utils\utils.py:57(bilinear_sampler)      
        591    1.604    0.003    1.604    0.003 {resize}
         99    1.335    0.013    1.335    0.013 {imwrite}
          1    0.092    0.092    1.332    1.332 c:\Users\tjerf\Desktop\Refactor_test\utils\blend\blender.py:18(_create_final_err_masks)     
    ```
