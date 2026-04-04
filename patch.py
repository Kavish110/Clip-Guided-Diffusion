import sys

content = open('diffusionclip.py', encoding='utf-8').read()
content = content.replace('["FFHQ", "AFHQ", "IMAGENET"]', '["FFHQ", "AFHQ", "IMAGENET", "TINY_IMAGENET"]')
content = content.replace('["FFHQ", "AFHQ"]', '["FFHQ", "AFHQ", "TINY_IMAGENET"]')
if 'import matplotlib.pyplot as plt' not in content:
    content = content.replace('import time', 'import time\nimport matplotlib.pyplot as plt\nimport json\nimport os')

# Inject loss accumulation, we need to find the print statement in train loop
# for clip_finetune:
target_print = 'print(\n                                f"CLIP {step}-{it_out}: loss_id: {loss_id:.3f}, loss_clip: {loss_clip:.3f}"\n                            )'
replacement_append = '''                            if not hasattr(self, 'loss_history'):
                                self.loss_history = {'clip': [], 'id': [], 'l1': [], 'total': []}
                            self.loss_history['clip'].append(loss_clip.item())
                            self.loss_history['id'].append(loss_id.item())
                            self.loss_history['l1'].append(loss_l1.item())
                            self.loss_history['total'].append(loss.item())
                            ''' + target_print

content = content.replace(target_print, replacement_append)

# for clip_finetune_eff:
target_print_eff = 'print(\n                                        f"CLIP {step}-{it_out}: loss_clip: {loss_clip:.3f}"\n                                    )'
replacement_append_eff = '''                                    if not hasattr(self, 'loss_history'):
                                        self.loss_history = {'clip': [], 'id': [], 'l1': [], 'total': []}
                                    self.loss_history['clip'].append(loss_clip.item())
                                    if self.config.data.dataset == "CelebA_HQ":
                                        self.loss_history['id'].append(loss_id.item())
                                    else:
                                        self.loss_history['id'].append(0)
                                    self.loss_history['l1'].append(loss_l1.item())
                                    self.loss_history['total'].append(loss.item())
                                    ''' + target_print_eff

content = content.replace(target_print_eff, replacement_append_eff)

# Add plotting logic at end of finetune
# where train loop ends:
train_end = 'scheduler_ft.step()'
plot_logic = '''scheduler_ft.step()
                        if hasattr(self, 'loss_history') and len(self.loss_history['total']) > 0:
                            plt.figure()
                            plt.plot(self.loss_history['clip'], label='CLIP Loss')
                            if sum(self.loss_history['id']) > 0:
                                plt.plot(self.loss_history['id'], label='ID Loss')
                            plt.plot(self.loss_history['l1'], label='L1 Loss')
                            plt.plot(self.loss_history['total'], label='Total Loss')
                            plt.legend()
                            plt.xlabel('Steps')
                            plt.ylabel('Loss')
                            plt.title('Training Losses')
                            plot_path = os.path.join(self.args.image_folder, f'training_loss.png')
                            plt.savefig(plot_path)
                            plt.close()
                            with open(os.path.join(self.args.image_folder, f'training_loss.json'), 'w') as f:
                                json.dump(self.loss_history, f)'''

content = content.replace(train_end, plot_logic)

open('diffusionclip.py', 'w', encoding='utf-8').write(content)
print("Patch OK")
