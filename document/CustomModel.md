<!--
 * @Author: Atoposiii zhenglei_zz@163.com
 * @Date: 2024-06-20 17:27:19
 * @LastEditors: Atoposiii zhenglei_zz@163.com
 * @LastEditTime: 2024-06-20 17:56:20
 * @FilePath: /ketnr/document/CustomModel.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE 
 -->

#### 1. 修改self.network

nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py

~~~python
self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
~~~
修改成self.network=ELCNet().to(self.device)


#### 2. 清除原网络深监督设置
self.enable_deep_supervision = False
